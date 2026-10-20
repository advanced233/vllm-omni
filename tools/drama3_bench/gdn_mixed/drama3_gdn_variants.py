# SPDX-License-Identifier: Apache-2.0
"""Instrumented copy of Drama 3's ``mixed_forward_core`` (vllm_omni ``fish_omni/gdn.py`` at da572cd).

``mixed_forward_core`` runs the GDN core of every step that holds prefills (30
calls per mixed step, one per backbone GDN layer). This copy keeps the installed
function's work line for line and adds two things:

* ``timer``: an optional :class:`LapTimer` that takes a ``perf_counter_ns``
  stamp after each section. It records host time only: nothing waits for the
  GPU, so a section's time is what the CPU spends issuing its work.
* ``direct_out``: on the non-FlashInfer path (A100), hand FLA the output rows
  ``core_attn_out[decode_tokens:tokens]`` as a flat view, so ``chunk_fwd_o``
  writes into them, instead of letting FLA allocate an output and copying it
  in afterwards. FLA's ``input_guard`` calls ``.contiguous()`` on every tensor
  argument, so a non-contiguous view would be written to a copy and lost; the
  rows are only handed over when they are contiguous, otherwise this falls
  back to the copy.

:func:`installed_matches` checks that the installed function is the one this
file copies, so a probe never measures a stale copy.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from typing import Any

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    causal_conv1d_fn,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from vllm_omni.model_executor.models.fish_omni.gdn import (
    _prefill_starts_int64,
    _uses_flashinfer_direct,
    zero_fresh_state_rows_,
)

# sha256 of the installed mixed_forward_core source (trailing spaces stripped) at da572cd.
EXPECTED_SOURCE_SHA256 = "9af0a98c23727bea4e7f34c1cf3283d838ad7a84a741b2fa8f946d823a4292a9"
SECTIONS = ("setup", "conv", "decode", "prep", "gather", "chunk", "scatter", "copy")


def source_sha256(fn: Any) -> str:
    src = inspect.getsource(fn)
    return hashlib.sha256("\n".join(line.rstrip() for line in src.strip().splitlines()).encode()).hexdigest()


def installed_matches() -> tuple[bool, str]:
    from vllm_omni.model_executor.models.fish_omni import gdn

    digest = source_sha256(gdn.mixed_forward_core)
    return digest == EXPECTED_SOURCE_SHA256, digest


class LapTimer:
    """Host-time laps of one call, in ns; ``laps`` keeps only the sections reached."""

    __slots__ = ("t0", "last", "laps")

    def __init__(self) -> None:
        self.t0 = self.last = time.perf_counter_ns()
        self.laps: dict[str, int] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter_ns()
        self.laps[name] = self.laps.get(name, 0) + now - self.last
        self.last = now

    def total(self) -> int:
        return self.last - self.t0


def mixed_forward_core(
    layer: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata: Any,
    *,
    direct_out: bool = False,
    timer: LapTimer | None = None,
) -> None:
    """``vllm_omni ... gdn.mixed_forward_core`` with optional laps and the direct output write."""
    md = attn_metadata
    kv_cache = layer.kv_cache
    conv_state = kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
    ssm_state = kv_cache[1]
    tokens = md.num_actual_tokens
    decode_tokens = md.num_decode_tokens
    mixed_qkv = mixed_qkv[:tokens]
    b = b[:tokens]
    a = a[:tokens]
    if timer is not None:
        timer.lap("setup")

    # 1. Causal convolution over decode and prefill tokens (updates every conv state once).
    conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    conv_out = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weights,
        layer.conv1d.bias,
        activation=layer.activation,
        conv_states=conv_state,
        has_initial_state=md.has_initial_state,
        cache_indices=md.non_spec_state_indices_tensor,
        query_start_loc=md.non_spec_query_start_loc,
        metadata=md,
    ).transpose(0, 1)
    if timer is not None:
        timer.lap("conv")

    # 2. Decode tokens: one token per request, state updated in place by the packed kernel.
    if decode_tokens > 0:
        packed = conv_out[:decode_tokens]
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=packed if packed.stride(-1) == 1 else packed.contiguous(),
            a=a[:decode_tokens],
            b=b[:decode_tokens],
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=core_attn_out[:decode_tokens].unsqueeze(1),
            ssm_state_indices=md.non_spec_state_indices_tensor[:decode_tokens],
            use_qk_l2norm_in_kernel=True,
        )
    if timer is not None:
        timer.lap("decode")

    # 3. Prefill tokens through the chunk kernel.
    direct = _uses_flashinfer_direct(layer, ssm_state)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out[decode_tokens:],
        a=a[decode_tokens:],
        b=b[decode_tokens:],
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads // layer.tp_size,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        apply_l2norm=True,
        output_g_exp=direct,
    )
    if timer is not None:
        timer.lap("prep")
    slots = md.prefill_state_indices
    if direct:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        zero_fresh_state_rows_(ssm_state, slots, md.prefill_has_initial_state)
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=_prefill_starts_int64(md),
            use_qk_l2norm_in_kernel=False,
            output=core_attn_out[decode_tokens:tokens],
            output_state=ssm_state,
            state_indices=slots,
            backend="flashinfer",
        )
        if timer is not None:
            timer.lap("chunk")
        return
    initial_state = ssm_state[slots]
    initial_state[~md.prefill_has_initial_state, ...] = 0
    if timer is not None:
        timer.lap("gather")
    extra: dict[str, torch.Tensor] = {}
    if direct_out:
        rows = core_attn_out[decode_tokens:tokens]
        if rows.is_contiguous():
            extra["core_attn_out"] = rows.view(-1)
    out, final_state = layer.chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=md.prefill_query_start_loc,
        chunk_indices=md.chunk_indices,
        chunk_offsets=md.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
        **extra,
    )
    if timer is not None:
        timer.lap("chunk")
    ssm_state[slots] = final_state.to(ssm_state.dtype)
    if timer is not None:
        timer.lap("scatter")
    if not extra:
        core_attn_out[decode_tokens:tokens] = out.squeeze(0)
    if timer is not None:
        timer.lap("copy")
