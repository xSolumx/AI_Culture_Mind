"""Execute architecture-level probes against maintained modern SSM libraries.

This is deliberately not a pretrained-quality benchmark.  It answers a
narrower question: can the actual maintained implementation be constructed,
run forward and backward, and produce finite tensors on the available runtime?
The JSON report keeps first-call compilation time separate from any systems
claim and records failures instead of silently substituting a local reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

SEED = 20_260_825


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError(f"unsupported probe output type: {type(output).__name__}")


def _probe_layer(
    factory: Callable[[], nn.Module],
    *,
    device: torch.device,
    training: bool = True,
    batch_size: int = 2,
    length: int = 64,
    hidden_size: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        module = factory().to(device=device, dtype=torch.float32)
        module.train(training)
        inputs = torch.randn(
            batch_size,
            length,
            hidden_size,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _unwrap_output(module(inputs))
        output.square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        return {
            "status": "pass",
            "module_training": training,
            "output_shape": list(output.shape),
            "parameters": _parameter_count(module),
            "finite_output": bool(torch.isfinite(output).all()),
            "finite_input_gradient": bool(
                inputs.grad is not None and torch.isfinite(inputs.grad).all()
            ),
            "finite_parameter_gradients": bool(
                gradients
                and all(torch.isfinite(gradient).all() for gradient in gradients)
            ),
            "parameters_with_gradients": len(gradients),
            "first_forward_backward_seconds": elapsed,
            "peak_cuda_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        }
    except Exception as error:  # noqa: BLE001 - fail-closed optional-backend probe
        return {
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _probe_causal_lm(
    factory: Callable[[], nn.Module],
    *,
    device: torch.device,
    vocab_size: int = 256,
    batch_size: int = 2,
    length: int = 32,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        module = factory().to(device=device, dtype=torch.float32).train()
        input_ids = torch.randint(0, vocab_size, (batch_size, length), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = module(input_ids=input_ids, labels=input_ids, use_cache=False)
        output = _unwrap_output(result)
        loss = getattr(result, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("causal LM probe did not return a tensor loss")
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        return {
            "status": "pass",
            "output_shape": list(output.shape),
            "parameters": _parameter_count(module),
            "loss": float(loss.detach()),
            "finite_output": bool(torch.isfinite(output).all()),
            "finite_parameter_gradients": bool(
                gradients
                and all(torch.isfinite(gradient).all() for gradient in gradients)
            ),
            "parameters_with_gradients": len(gradients),
            "first_forward_backward_seconds": elapsed,
            "peak_cuda_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        }
    except Exception as error:  # noqa: BLE001 - fail-closed optional-backend probe
        return {
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def probe_fla(device: torch.device, only: str | None = None) -> dict[str, Any]:
    import fla
    from fla.layers import (
        GatedDeltaNet,
        GatedDeltaNet2,
        GatedDeltaProduct,
        KimiDeltaAttention,
        Mamba3,
    )

    factories: dict[str, Callable[[], nn.Module]] = {
        "fla_gated_deltanet": lambda: GatedDeltaNet(
            hidden_size=64,
            expand_v=1,
            head_dim=16,
            num_heads=4,
            mode="fused_recurrent",
            use_short_conv=False,
        ),
        "fla_gated_deltanet_2": lambda: GatedDeltaNet2(
            hidden_size=64,
            expand_v=1,
            head_dim=16,
            num_heads=4,
            mode="fused_recurrent",
            use_short_conv=False,
        ),
        "fla_kimi_delta_attention": lambda: KimiDeltaAttention(
            hidden_size=64,
            expand_v=1,
            head_dim=16,
            num_heads=4,
            mode="fused_recurrent",
            use_short_conv=False,
        ),
        "fla_gated_delta_product": lambda: GatedDeltaProduct(
            hidden_size=64,
            expand_v=1,
            head_dim=16,
            num_heads=4,
            mode="fused_recurrent",
            use_short_conv=False,
            num_householder=2,
        ),
        "fla_mamba3_siso": lambda: Mamba3(
            hidden_size=64,
            state_size=16,
            expand=2,
            head_dim=16,
            n_groups=1,
            is_mimo=False,
            chunk_size=64,
        ),
        "fla_mamba3_mimo": lambda: Mamba3(
            hidden_size=64,
            state_size=16,
            expand=2,
            head_dim=16,
            n_groups=1,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=64,
        ),
    }
    return {
        "backend": "flash-linear-attention",
        "backend_version": getattr(fla, "__version__", "unknown"),
        "implementation_scope": (
            "actual maintained FLA layer implementations in recurrent eval mode "
            "with autograd; random tiny architecture probes, not supported chunk "
            "training or pretrained checkpoint quality"
        ),
        "probes": {
            name: _probe_layer(factory, device=device, training=False)
            for name, factory in factories.items()
            if only is None or name == only
        },
    }


def probe_transformers(device: torch.device, only: str | None = None) -> dict[str, Any]:
    import transformers
    from transformers import (
        FalconH1Config,
        FalconH1ForCausalLM,
        Mamba2Config,
        Mamba2ForCausalLM,
        NemotronHConfig,
        NemotronHForCausalLM,
        Qwen3NextConfig,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    qwen_config = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        layer_types=["linear_attention"],
    )

    model_factories: dict[str, Callable[[], nn.Module]] = {
        "transformers_mamba2_lm": lambda: Mamba2ForCausalLM(
            Mamba2Config(
                vocab_size=256,
                hidden_size=64,
                num_hidden_layers=2,
                num_heads=8,
                head_dim=16,
                state_size=16,
                expand=2,
                conv_kernel=4,
                n_groups=1,
                chunk_size=16,
                residual_in_fp32=False,
            )
        ),
        "transformers_falcon_h1_lm": lambda: FalconH1ForCausalLM(
            FalconH1Config(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                mamba_d_ssm=64,
                mamba_n_heads=4,
                mamba_d_head=16,
                mamba_n_groups=1,
                mamba_d_state=16,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_chunk_size=16,
                max_position_embeddings=128,
            )
        ),
        "transformers_nemotron_h_lm": lambda: NemotronHForCausalLM(
            NemotronHConfig(
                vocab_size=256,
                hidden_size=64,
                layers_block_type=["mamba", "attention", "mlp"],
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                intermediate_size=128,
                use_mamba_kernels=False,
                ssm_state_size=16,
                mamba_num_heads=4,
                mamba_head_dim=16,
                n_groups=1,
                conv_kernel=4,
                expand=2,
                chunk_size=16,
                max_position_embeddings=128,
                n_routed_experts=1,
                n_shared_experts=1,
                num_experts_per_tok=1,
                moe_intermediate_size=64,
                moe_shared_expert_intermediate_size=64,
            )
        ),
    }
    return {
        "backend": "transformers",
        "backend_version": transformers.__version__,
        "implementation_scope": (
            "actual maintained Transformers classes; random tiny architecture "
            "probes, not pretrained checkpoint quality"
        ),
        "probes": {
            **(
                {
                    "transformers_qwen3_next_gdn_layer": _probe_layer(
                        lambda: Qwen3NextGatedDeltaNet(qwen_config, layer_idx=0),
                        device=device,
                    )
                }
                if only in (None, "transformers_qwen3_next_gdn_layer")
                else {}
            ),
            **{
                name: _probe_causal_lm(factory, device=device)
                for name, factory in model_factories.items()
                if only is None or name == only
            },
        },
    }


def build_report(
    backend: str, device_name: str, only: str | None = None
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    backend_report = (
        probe_fla(device, only)
        if backend == "fla"
        else probe_transformers(device, only)
    )
    return {
        "schema_version": 1,
        "seed": SEED,
        "claim_boundary": (
            "availability and finite forward/backward only; no pretrained quality, "
            "throughput, scaling, or model-ranking claim"
        ),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        **backend_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transformers", "fla"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--only")
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_report(arguments.backend, arguments.device, arguments.only), indent=2
        )
    )


if __name__ == "__main__":
    main()
