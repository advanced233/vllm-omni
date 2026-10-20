"""Supplement only: use installed vLLM's actual Triton path in the original GDN fixture.

The repository fixture binds fi_chunk_gated_delta_rule for both backend labels.
This plugin substitutes only that callable with the production forward_native
method (which calls vLLM's vendored FLA/Triton implementation). The source tests,
inputs, state dtypes, seeds, assertions, tolerances and mixed/standard paths are
unchanged. Original unmodified failures remain in pytest.log and junit.xml.
"""
import pytest


@pytest.fixture(autouse=True)
def actual_triton_fixture(request, monkeypatch):
    if request.module.__name__ != "tests.model_executor.models.fish_omni.test_fish_omni_gdn":
        return
    if not request.node.name.startswith("test_drama3_gdn_core_matches_standard_path["):
        return
    assert request.node.callspec.params["prefill_backend"] == "triton"
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule

    original_layer = request.module._layer

    def make_layer(device, pools):
        layer, dim = original_layer(device, pools)
        # Exactly the installed production Triton adapter, including all kwargs.
        layer.chunk_gated_delta_rule._fn = lambda **kw: ChunkGatedDeltaRule.forward_native(None, **kw)
        layer.chunk_gated_delta_rule.gdn_prefill_backend = "triton"
        return layer, dim

    monkeypatch.setattr(request.module, "_layer", make_layer)
    print("SUPPLEMENTAL: GDN fixture callable -> installed ChunkGatedDeltaRule.forward_native (FLA/Triton)")
