"""Immutable source pins and honest comparison boundaries for external models."""

from __future__ import annotations

SCHEMA_VERSION = 1

GITHUB_SOURCES = {
    "mamba": {
        "url": "https://github.com/state-spaces/mamba.git",
        "revision": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
    },
    "hybrid_model_factory": {
        "url": "https://github.com/awslabs/hybrid-model-factory.git",
        "revision": "04b6242b8a87e4496e4ade65eb2ec6a8b23b262a",
    },
    "flash_linear_attention": {
        "url": "https://github.com/fla-org/flash-linear-attention.git",
        "revision": "27967b970eaaf982a6960abf6cba8add9c34c7cc",
    },
}

# These are checkpoint byte counts reported by the Hugging Face Hub API at the
# exact revisions below. They are not estimates of runtime memory, which also
# includes activations, recurrent/KV state, workspaces, and allocator overhead.
BASELINES = (
    {
        "id": "falcon_mamba_7b",
        "family": "Falcon Mamba",
        "variant": "7B base",
        "hf_repo": "tiiuae/falcon-mamba-7b",
        "revision": "080ad94b3619e2c2d0afa59bafdc6113465b7006",
        "weight_bytes": 14_545_401_832,
        "implementation": "transformers.FalconMambaForCausalLM",
        "comparison_tiers": ("native_checkpoint_inference",),
        "local_plan": "quantized_or_cpu_offload_only",
        "boundary": (
            "Different tokenizer, training corpus, scale, and license; never mix "
            "its pretrained loss with the from-scratch Shakespeare table."
        ),
    },
    {
        "id": "mamba3_siso_187m",
        "family": "Mamba-3",
        "variant": "SISO 187M",
        "hf_repo": "state-spaces/mamba3-siso-187m",
        "revision": "6792c27c00f3bb41506db1066dcd1c51bb0f4b02",
        "weight_bytes": 373_744_619,
        "implementation": "state-spaces/mamba Mamba3(is_mimo=False)",
        "comparison_tiers": (
            "matched_shakespeare_training",
            "native_checkpoint_inference",
        ),
        "local_plan": "separate_mamba3_environment",
        "boundary": (
            "The matched layer must be trained from scratch; the official 187M "
            "checkpoint belongs only in native-tokenizer inference tests."
        ),
    },
    {
        "id": "mamba3_mimo_187m",
        "family": "Mamba-3",
        "variant": "MIMO 187M",
        "hf_repo": "state-spaces/mamba3-mimo-187m",
        "revision": "8fd6e9eb7b795f2e15d7f6353251d0137980c43e",
        "weight_bytes": 374_640_827,
        "implementation": "state-spaces/mamba Mamba3(is_mimo=True)",
        "comparison_tiers": (
            "matched_shakespeare_training",
            "native_checkpoint_inference",
        ),
        "local_plan": "separate_tilelang_mamba3_environment",
        "boundary": (
            "MIMO uses a TileLang kernel and must not add or upgrade packages in "
            "the frozen Torch 2.10 Mamba-2 environment."
        ),
    },
    {
        "id": "gka_primed_qwen3_8b",
        "family": "Gated KalmaNet",
        "variant": "50% GKA primed HQwen3 8B",
        "hf_repo": "amazon/GKA-primed-HQwen3-8B-Instruct",
        "revision": "497c163ef7757b1df66f9a63ca28a603a8aa0ab5",
        "weight_bytes": 17_000_556_608,
        "implementation": "awslabs/hybrid-model-factory",
        "comparison_tiers": (
            "matched_shakespeare_training",
            "remote_native_checkpoint_reference",
        ),
        "local_plan": "small_layer_only_on_turing",
        "boundary": (
            "The official 8B BF16 checkpoint exceeds local VRAM and published "
            "factory tests target A100/H200; only a separately validated small "
            "layer belongs in a local matched run."
        ),
    },
    {
        "id": "gdn_primed_qwen3_8b",
        "family": "Gated DeltaNet",
        "variant": "50% GDN primed HQwen3 8B",
        "hf_repo": "amazon/GDN-primed-HQwen3-8B-Instruct",
        "revision": "4324333ef4a23f39a6a15f50b8548c14490056cf",
        "weight_bytes": 16_995_829_952,
        "implementation": "awslabs/hybrid-model-factory",
        "comparison_tiers": (
            "matched_shakespeare_training",
            "remote_native_checkpoint_reference",
        ),
        "local_plan": "small_layer_only_on_turing",
        "boundary": (
            "The official 8B BF16 checkpoint exceeds local VRAM; an FLA or "
            "factory layer must first pass Turing forward/backward parity."
        ),
    },
    {
        "id": "jamba_v0_1",
        "family": "AI21 Jamba",
        "variant": "v0.1 12B-active/52B-total",
        "hf_repo": "ai21labs/Jamba-v0.1",
        "revision": "9efd11575ba791d9e3d25d4c8b670e78506b2df7",
        "weight_bytes": 103_148_144_224,
        "implementation": "transformers.JambaForCausalLM",
        "comparison_tiers": ("remote_native_checkpoint_reference",),
        "local_plan": "metadata_only",
        "boundary": (
            "The original requested Jamba lineage is far beyond an 8 GB GPU; "
            "paper/model-card numbers remain external references, not local runs."
        ),
    },
    {
        "id": "jamba2_3b",
        "family": "AI21 Jamba",
        "variant": "Jamba2 3B",
        "hf_repo": "ai21labs/AI21-Jamba2-3B",
        "revision": "525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9",
        "weight_bytes": 6_394_271_296,
        "implementation": "transformers.JambaForCausalLM",
        "comparison_tiers": ("native_checkpoint_inference",),
        "local_plan": "quantized_or_cpu_offload_validation_first",
        "boundary": (
            "BF16 weights alone consume most local VRAM; runtime feasibility "
            "must be proved before recording throughput."
        ),
    },
)
