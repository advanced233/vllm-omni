# SPDX-License-Identifier: Apache-2.0
"""One Drama 3 GDN layer, its state pools and a mixed step's metadata, on the real FLA/Triton path.

Built like ``tests/model_executor/models/fish_omni/test_fish_omni_gdn.py``
(same head sizes, same layer stand-in), with two differences that make it
what an A100 runs:

* the layer's ``chunk_gated_delta_rule`` is vLLM's production Triton/FLA adapter
  (``ChunkGatedDeltaRule.forward_native``), not FlashInfer: the upstream fixture
  binds FlashInfer under both backend labels;
* the metadata carries the FLA chunk indices / offsets precomputed on the CPU,
  as ``GDNAttentionMetadataBuilder`` does for the non-FlashInfer backend, so
  the chunk kernel does not compute them (with a device-to-host copy) itself.

A step is written ``"D:P"``: ``D`` decode requests of one token, then the
prefills ``P``, comma-separated ``<tokens><f|c>`` (``f`` fresh: no initial state,
on a pool row that still holds a previous request's state; ``c`` continued).
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

K_HEADS, V_HEADS, HEAD_K, HEAD_V, WIDTH = 16, 32, 128, 128, 4
SPARE_ROWS = 4  # pool rows no request uses: must come out bit for bit unchanged

# Mixed steps shaped like the A100 profiles (decode requests, prefill tokens) at C8 / C16 / C64.
PRESETS = {
    "c8": "7:170f",
    "c16": "15:176f",
    "c64": "60:180f,178f",
    "upstream": "3:37f,70c",
    "prefill_only": "0:37f,70c",
    "long_continued": "8:512c",
}


@dataclass
class Step:
    decodes: int
    prefills: list[tuple[int, bool]]  # (tokens, has_initial_state)

    @property
    def tokens(self) -> int:
        return self.decodes + sum(n for n, _ in self.prefills)

    @property
    def rows(self) -> int:
        # Row 0 is vLLM's NULL_BLOCK_ID: the scheduler never gives it to a request and the kernels skip it.
        return 1 + self.decodes + len(self.prefills) + SPARE_ROWS


def parse_step(spec: str) -> Step:
    spec = PRESETS.get(spec, spec)
    decodes, _, prefills = spec.partition(":")
    parsed = []
    for item in filter(None, prefills.split(",")):
        kind = item[-1]
        if kind not in "fc":
            raise ValueError(f"prefill {item!r}: end it with f (fresh) or c (continued)")
        parsed.append((int(item[:-1]), kind == "c"))
    if not parsed:
        raise ValueError(f"{spec!r}: a mixed step needs at least one prefill")
    return Step(int(decodes), parsed)


class CallableNamespace:
    def __init__(self, fn, **attrs) -> None:
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, **kw):
        return self._fn(**kw)


def make_pools(device: torch.device, rows: int, state_dtype: torch.dtype, seed: int = 1):
    from vllm.model_executor.layers.mamba import mamba_utils

    conv_shape, ssm_shape = mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, K_HEADS, V_HEADS, HEAD_K, HEAD_V, WIDTH
    )
    g = torch.Generator(device="cpu").manual_seed(seed)
    conv = (torch.randn(rows, *conv_shape, generator=g) * 0.5).to(device, torch.bfloat16)
    ssm = (torch.randn(rows, *ssm_shape, generator=g) * 0.05).to(device, state_dtype)
    return conv, ssm


def make_layer(device: torch.device, pools, seed: int = 0):
    """The upstream test's layer stand-in, with the production FLA/Triton chunk adapter."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
        QwenGatedDeltaNetAttention,
    )

    dim = 2 * K_HEADS * HEAD_K + V_HEADS * HEAD_V
    g = torch.Generator(device="cpu").manual_seed(seed)
    layer = types.SimpleNamespace(
        kv_cache=pools,
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(dim, 1, WIDTH, generator=g) * 0.3).to(device, torch.bfloat16),
            bias=(torch.randn(dim, generator=g) * 0.1).to(device, torch.bfloat16),
        ),
        activation="silu",
        A_log=torch.randn(V_HEADS, generator=g).to(device),
        dt_bias=torch.randn(V_HEADS, generator=g).to(device),
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        tp_size=1,
        head_k_dim=HEAD_K,
        head_v_dim=HEAD_V,
        key_dim=K_HEADS * HEAD_K,
        value_dim=V_HEADS * HEAD_V,
        prefix="gdn",
        enable_packed_recurrent_decode=False,
    )
    # The adapter A100 runs: forward_native calls vLLM's vendored FLA chunk kernels.
    layer.chunk_gated_delta_rule = CallableNamespace(
        lambda **kw: ChunkGatedDeltaRule.forward_native(None, **kw), gdn_prefill_backend="triton"
    )
    layer.rearrange_mixed_qkv = types.MethodType(QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer)
    return layer, dim


def slots_for(step: Step) -> list[int]:
    """Decodes on rows 1..D, prefills on the next rows (all holding stale state); the rest stay unused."""
    return list(range(1, 1 + step.decodes + len(step.prefills)))


def unused_rows(step: Step, slots: list[int] | None = None) -> list[int]:
    """Pool rows the step must leave untouched: the null block and the spare rows."""
    used = set(slots_for(step) if slots is None else slots)
    return [r for r in range(step.rows) if r not in used]


def make_metadata(device: torch.device, step: Step, slots: list[int] | None = None):
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, compute_causal_conv1d_metadata

    slots = slots_for(step) if slots is None else slots
    if NULL_BLOCK_ID in slots:
        raise ValueError(f"slot {NULL_BLOCK_ID} is vLLM's null block, never a request's state row")
    d = step.decodes
    lengths = [1] * d + [n for n, _ in step.prefills]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    has_initial = torch.tensor([True] * d + [h for _, h in step.prefills], device=device)
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(starts, device=device)
    slots_t = torch.tensor(slots, dtype=torch.int32, device=device)
    starts_dev = starts.to(device)
    prefill_starts_cpu = starts[d:] - d
    return GDNAttentionMetadata(
        num_prefills=len(step.prefills),
        num_prefill_tokens=int(starts[-1]) - d,
        num_decodes=d,
        num_decode_tokens=d,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(starts[-1]),
        has_initial_state=has_initial,
        non_spec_query_start_loc=starts_dev,
        non_spec_state_indices_tensor=slots_t,
        prefill_query_start_loc=starts_dev[d:] - d,
        prefill_state_indices=slots_t[d:],
        prefill_has_initial_state=has_initial[d:],
        # As GDNAttentionMetadataBuilder._build_chunk_metadata does for the Triton backend.
        chunk_indices=prepare_chunk_indices(prefill_starts_cpu, FLA_CHUNK_SIZE).to(device),
        chunk_offsets=prepare_chunk_offsets(prefill_starts_cpu, FLA_CHUNK_SIZE).to(device),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offset_ptr,
    )


def make_inputs(device: torch.device, tokens: int, dim: int, seed: int = 2, pad: int = 3, sentinel: float = 7.0):
    """mixed_qkv / b / a of a step and an output buffer whose ``pad`` rows past the step hold ``sentinel``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mixed_qkv = (torch.randn(tokens, dim, generator=g) * 0.5).to(device, torch.bfloat16)
    b = torch.randn(tokens, V_HEADS, generator=g).to(device, torch.bfloat16)
    a = torch.randn(tokens, V_HEADS, generator=g).to(device, torch.bfloat16)
    out = torch.full((tokens + pad, V_HEADS, HEAD_V), sentinel, device=device, dtype=torch.bfloat16)
    return mixed_qkv, b, a, out
