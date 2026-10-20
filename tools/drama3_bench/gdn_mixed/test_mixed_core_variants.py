# SPDX-License-Identifier: Apache-2.0
"""The instrumented copy and the direct-output variant of Drama 3's mixed GDN core, on the FLA/Triton path.

* Both variants must equal the installed ``mixed_forward_core`` bit for bit:
  outputs, conv pool and SSM pool. They run the same kernels on the same
  inputs; only where FLA writes its output differs.
* They must also match vLLM's standard ``_forward_core`` within the upstream
  test's tolerances, on steps shaped like the A100 profiles.
* Rows past the step in the output buffer, and pool rows no request uses
  (the null block, row 0, among them), must come out unchanged; a
  non-contiguous output buffer must take the copy.
* A prompt split over two steps (fresh, then continued on the same row) must
  agree after both steps.

Run from the vllm-omni checkout:
  python -m pytest -v tools/drama3_bench/gdn_mixed/test_mixed_core_variants.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drama3_gdn_harness as harness  # noqa: E402

CASES = ["upstream", "prefill_only", "c16", "c64", "long_continued"]
STATE_DTYPES = [torch.float32, torch.bfloat16]
SENTINEL = 7.0


def _installed():
    from vllm_omni.model_executor.models.fish_omni.gdn import mixed_forward_core

    return mixed_forward_core


def _variant(direct_out: bool):
    import drama3_gdn_variants as variants

    def fn(layer, mixed_qkv, b, a, out, md):
        variants.mixed_forward_core(layer, mixed_qkv, b, a, out, md, direct_out=direct_out)

    return fn


VARIANTS = {"baseline": lambda: _variant(False), "direct_out": lambda: _variant(True)}


def _prefixed(name: str):
    # A callable msg keeps assert_close's mismatch details; a string would replace them.
    return lambda detail: f"{name}: {detail}"


def _run(fn, spec: str, state_dtype: torch.dtype, *, out_layout=None, before_call=None):
    device = torch.device("cuda")
    step = harness.parse_step(spec)
    conv0, ssm0 = harness.make_pools(device, step.rows, state_dtype)
    pools = (conv0.clone(), ssm0.clone())
    layer, dim = harness.make_layer(device, pools)
    md = harness.make_metadata(device, step)
    mixed_qkv, b, a, out = harness.make_inputs(device, step.tokens, dim, sentinel=SENTINEL)
    if out_layout is not None:
        out = out_layout(out)
    if before_call is not None:
        before_call(md)
    fn(layer, mixed_qkv, b, a, out, md)
    torch.cuda.synchronize()
    return out, pools, (conv0, ssm0), step


def test_copy_is_the_installed_version():
    import drama3_gdn_variants as variants

    matches, digest = variants.installed_matches()
    assert matches, f"installed mixed_forward_core sha256 {digest} != copied {variants.EXPECTED_SOURCE_SHA256}"


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_equals_installed_bit_for_bit(variant, case, state_dtype):
    ref_out, ref_pools, (conv0, ssm0), step = _run(_installed(), case, state_dtype)
    out, pools, _, _ = _run(VARIANTS[variant](), case, state_dtype)

    # NaN != NaN would hide which side went wrong: check finiteness first.
    assert torch.isfinite(ref_out[: step.tokens]).all(), "installed output not finite"
    assert torch.isfinite(out[: step.tokens]).all(), f"{variant} output not finite"
    assert torch.equal(out, ref_out)
    assert torch.equal(pools[0], ref_pools[0])
    assert torch.equal(pools[1], ref_pools[1])
    # Output rows past the step are not written.
    assert torch.all(out[step.tokens :] == SENTINEL)
    # Pool rows no request uses keep their state.
    for row in harness.unused_rows(step):
        assert torch.equal(pools[0][row], conv0[row])
        assert torch.equal(pools[1][row], ssm0[row])


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("case", CASES)
def test_variants_match_vllm_standard_path(monkeypatch, case, state_dtype):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_module
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    def standard(layer, mixed_qkv, b, a, out, _md):
        QwenGatedDeltaNetAttention._forward_core(layer, mixed_qkv, b, a, out)

    def bind(md):
        monkeypatch.setattr(gdn_module, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata={"gdn": md}))

    ref_out, ref_pools, _, step = _run(standard, case, state_dtype, before_call=bind)
    tol = 2e-3 if state_dtype == torch.float32 else 1e-2  # one bf16 ulp on a bf16 pool, as upstream
    for name, make in VARIANTS.items():
        out, pools, _, _ = _run(make(), case, state_dtype)
        msg = _prefixed(name)
        torch.testing.assert_close(
            out[: step.tokens].float(), ref_out[: step.tokens].float(), atol=2e-2, rtol=2e-2, msg=msg
        )
        torch.testing.assert_close(pools[0], ref_pools[0], msg=msg)
        torch.testing.assert_close(pools[1].float(), ref_pools[1].float(), atol=tol, rtol=tol, msg=msg)


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=["fp32", "bf16"])
def test_non_contiguous_output_takes_the_copy(state_dtype):
    # No decodes: the packed decode kernel never sees the strided buffer.
    def strided(out):
        base = torch.full((out.shape[1], out.shape[0], out.shape[2]), SENTINEL, device=out.device, dtype=out.dtype)
        view = base.transpose(0, 1)
        assert view.shape == out.shape and not view[: out.shape[0] - 3].is_contiguous()
        return view

    ref_out, ref_pools, _, step = _run(_installed(), "prefill_only", state_dtype, out_layout=strided)
    out, pools, _, _ = _run(_variant(True), "prefill_only", state_dtype, out_layout=strided)
    assert torch.equal(out, ref_out)
    assert torch.equal(pools[1], ref_pools[1])
    assert torch.all(out[step.tokens :] == SENTINEL)


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prompt_split_over_two_steps(variant, state_dtype):
    """Step 1 starts a prompt on row 3 (fresh, stale row); step 2 continues it on the same row."""
    device = torch.device("cuda")
    first, second = harness.parse_step("2:96f"), harness.parse_step("2:80c")
    rows = first.rows

    def run(fn):
        conv0, ssm0 = harness.make_pools(device, rows, state_dtype)
        pools = (conv0.clone(), ssm0.clone())
        layer, dim = harness.make_layer(device, pools)
        outs = []
        for i, step in enumerate((first, second)):
            md = harness.make_metadata(device, step, slots=[1, 2, 3])
            mixed_qkv, b, a, out = harness.make_inputs(device, step.tokens, dim, seed=10 + i, sentinel=SENTINEL)
            fn(layer, mixed_qkv, b, a, out, md)
            outs.append(out)
        torch.cuda.synchronize()
        return outs, pools

    ref_outs, ref_pools = run(_installed())
    outs, pools = run(VARIANTS[variant]())
    for out, ref in zip(outs, ref_outs, strict=True):
        assert torch.equal(out, ref)
    assert torch.equal(pools[0], ref_pools[0])
    assert torch.equal(pools[1], ref_pools[1])
