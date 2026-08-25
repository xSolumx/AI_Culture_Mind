"""Fail-closed probes for native SM75 kernels used by v1.4 research.

Each invocation runs one maintained implementation.  No local recurrence,
Transformers eager path, or PyTorch attention result can substitute for a
missing backend.  The reference attention is used only for numerical error
measurement after the Turing CUDA extension has executed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from torch import nn
from torch.nn import functional as F

SEED = 20_260_825
FLASH_OUTPUT_MAX_ABS_TOLERANCE = 1.0e-3
FLASH_GRADIENT_MAX_ABS_TOLERANCE = 2.0e-3
SOURCE_ROOT: Path | None = None


def _git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime() -> dict[str, Any]:
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else None,
        "compute_capability": list(capability) if capability is not None else None,
    }


def _require_sm75() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise RuntimeError(f"probe requires SM75, found {capability}")
    return torch.device("cuda")


def _source_install_provenance(
    distribution_name: str,
    module: ModuleType,
    source_root: Path,
) -> dict[str, Any]:
    root = source_root.resolve()
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{distribution_name} has no direct_url.json provenance")
    direct_url = json.loads(direct_url_text)
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        raise RuntimeError(
            f"{distribution_name} is not installed from a local source tree"
        )
    installed_from = Path(unquote(parsed.path)).resolve()
    if installed_from != root:
        raise RuntimeError(
            f"{distribution_name} was installed from {installed_from}, not {root}"
        )
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise TypeError(f"{module.__name__} has no resolved module file")
    return {
        "distribution": distribution_name,
        "distribution_version": distribution.version,
        "direct_url": direct_url,
        "module": module.__name__,
        "module_file": str(Path(module_file).resolve()),
        "source_root": str(root),
        "source_revision": _git_revision(root),
    }


def _finite_layer_probe(
    factory: Callable[[], nn.Module],
    *,
    training: bool,
    length: int = 64,
    hidden_size: int = 64,
) -> dict[str, Any]:
    device = _require_sm75()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    module = factory().to(device=device, dtype=torch.float32)
    module.train(training)
    inputs = torch.randn(
        1,
        length,
        hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = module(inputs)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("backend did not return a tensor")
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    loss = output.float().square().mean()
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    trainable_named = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    gradients = [
        parameter.grad for _, parameter in trainable_named if parameter.grad is not None
    ]
    missing_gradient_names = [
        name for name, parameter in trainable_named if parameter.grad is None
    ]
    nonfinite_gradient_names = [
        name
        for name, parameter in trainable_named
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    finite_output = bool(torch.isfinite(output).all())
    finite_input_gradient = bool(
        inputs.grad is not None and torch.isfinite(inputs.grad).all()
    )
    complete_parameter_gradients = bool(trainable_named) and not missing_gradient_names
    finite_parameter_gradients = complete_parameter_gradients and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if not (
        finite_output
        and finite_input_gradient
        and complete_parameter_gradients
        and finite_parameter_gradients
    ):
        raise RuntimeError(
            "native layer failed finiteness or complete-gradient qualification: "
            f"output={finite_output}, input_grad={finite_input_gradient}, "
            f"parameter_grads={len(gradients)}/{len(trainable_named)}, "
            f"finite_parameter_grads={finite_parameter_gradients}, "
            f"missing={missing_gradient_names}, nonfinite={nonfinite_gradient_names}"
        )
    return {
        "status": "pass",
        "module_training": training,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "finite_output": finite_output,
        "finite_input_gradient": finite_input_gradient,
        "complete_parameter_gradients": complete_parameter_gradients,
        "finite_parameter_gradients": finite_parameter_gradients,
        "parameters_with_gradients": len(gradients),
        "trainable_parameter_tensors": len(trainable_named),
        "forward_seconds_first_call": forward_seconds,
        "backward_seconds_first_call": time.perf_counter() - backward_started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    }


def _gdn2(*, chunk: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    import fla
    from fla.layers.gdn2 import GatedDeltaNet2

    if SOURCE_ROOT is None:
        raise RuntimeError("GDN2 probe requires source-root provenance")

    mode = "chunk" if chunk else "fused_recurrent"
    probe = _finite_layer_probe(
        lambda: GatedDeltaNet2(
            hidden_size=64,
            expand_v=1.0,
            head_dim=16,
            num_heads=4,
            mode=mode,
            use_short_conv=chunk,
        ),
        training=chunk,
    )
    eligibility = {
        "training_baseline_eligible": chunk,
        "inference_gradient_control_eligible": True,
        "reason": (
            "native chunk training forward/backward passed"
            if chunk
            else "native recurrent eval/autograd passed, but FLA forbids this mode in module.train()"
        ),
    }
    return {
        "implementation": "fla.layers.gdn2.GatedDeltaNet2",
        "implementation_version": getattr(fla, "__version__", "unknown"),
        "mode": mode,
        "source_provenance": _source_install_provenance(
            "flash-linear-attention", fla, SOURCE_ROOT
        ),
        **probe,
    }, eligibility


def _mamba3(*, mimo: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    import mamba_ssm
    from mamba_ssm import Mamba3

    if SOURCE_ROOT is None:
        raise RuntimeError("Mamba-3 probe requires source-root provenance")

    rank = 2
    probe = _finite_layer_probe(
        lambda: Mamba3(
            d_model=64,
            d_state=16,
            expand=2,
            headdim=32 if mimo else 16,
            is_mimo=mimo,
            mimo_rank=rank,
            chunk_size=64 // rank if mimo else 64,
            device="cuda",
            dtype=torch.float32,
        ),
        training=True,
    )
    return {
        "implementation": "mamba_ssm.Mamba3",
        "implementation_version": getattr(mamba_ssm, "__version__", "unknown"),
        "is_mimo": mimo,
        "source_provenance": _source_install_provenance(
            "mamba-ssm", mamba_ssm, SOURCE_ROOT
        ),
        **probe,
    }, {
        "training_baseline_eligible": True,
        "reason": "native maintained Mamba-3 training forward/backward passed on SM75",
    }


def _attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
    ).transpose(1, 2)


def _flash_turing(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sm75()
    # Resolve the installed extension and its PEP 610 metadata before adding
    # the source checkout for the Python interface. The checkout contains
    # build metadata without direct_url.json and must not shadow the installed
    # distribution during provenance qualification.
    import flash_attn_turing

    extension_provenance = _source_install_provenance(
        "flash-attn-turing", flash_attn_turing, source_root
    )
    sys.path.insert(0, str(source_root))
    import flash_attention_interface
    from flash_attention_interface import flash_attn_func

    cases = []
    for head_dim in (64, 128):
        for causal in (False, True):
            torch.manual_seed(SEED + head_dim + int(causal))
            tensors = [
                torch.randn(
                    1,
                    128,
                    2,
                    head_dim,
                    device="cuda",
                    dtype=torch.float16,
                    requires_grad=True,
                )
                for _ in range(3)
            ]
            q, k, v = tensors
            torch.cuda.synchronize()
            started = time.perf_counter()
            actual = flash_attn_func(q, k, v, causal=causal)
            upstream = torch.randn_like(actual)
            actual.backward(upstream)
            torch.cuda.synchronize()
            actual_grads = [tensor.grad.detach().clone() for tensor in tensors]

            refs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
            expected = _attention_reference(*refs, causal=causal)
            expected.backward(upstream)
            cases.append(
                {
                    "head_dim": head_dim,
                    "causal": causal,
                    "finite_output": bool(torch.isfinite(actual).all()),
                    "finite_gradients": all(
                        bool(torch.isfinite(gradient).all())
                        for gradient in actual_grads
                    ),
                    "output_max_abs_error": float(
                        (actual.float() - expected.float()).abs().max().detach()
                    ),
                    "gradient_max_abs_errors": [
                        float(
                            (actual_grad.float() - reference.grad.float())
                            .abs()
                            .max()
                            .detach()
                        )
                        for actual_grad, reference in zip(
                            actual_grads, refs, strict=True
                        )
                    ],
                    "forward_backward_seconds_first_call": time.perf_counter()
                    - started,
                }
            )
    accepted = all(
        case["finite_output"]
        and case["finite_gradients"]
        and case["output_max_abs_error"] <= FLASH_OUTPUT_MAX_ABS_TOLERANCE
        and max(case["gradient_max_abs_errors"]) <= FLASH_GRADIENT_MAX_ABS_TOLERANCE
        for case in cases
    )
    if not accepted:
        raise RuntimeError("Turing FlashAttention exceeded its frozen numerical gate")
    return {
        "implementation": "ssiu/flash-attention-turing",
        "extension": "flash_attn_turing",
        "numerical_gate": {
            "output_max_abs_tolerance": FLASH_OUTPUT_MAX_ABS_TOLERANCE,
            "gradient_max_abs_tolerance": FLASH_GRADIENT_MAX_ABS_TOLERANCE,
        },
        "source_provenance": {
            "interface_file": str(Path(flash_attention_interface.__file__).resolve()),
            "extension_install": extension_provenance,
        },
        "status": "pass",
        "cases": cases,
    }, {
        "training_baseline_eligible": True,
        "reason": "native sm_75 extension forward/backward passed against an SDPA numerical reference",
    }


def main() -> int:
    global SOURCE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=(
            "gdn2-recurrent",
            "gdn2-chunk",
            "mamba3-siso",
            "mamba3-mimo",
            "flash-turing",
        ),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend != "flash-turing" and args.source_root is None:
        parser.error(f"{args.backend} requires --source-root for provenance")
    SOURCE_ROOT = args.source_root.resolve() if args.source_root is not None else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "backend": args.backend,
        "claim_boundary": (
            "native maintained implementation and finite forward/backward only; "
            "first-call time is compilation-inclusive and not a throughput result"
        ),
        "runtime": _runtime(),
        "source_root": str(args.source_root) if args.source_root else None,
        "source_revision": _git_revision(args.source_root),
    }
    try:
        if args.backend == "gdn2-recurrent":
            probe, eligibility = _gdn2(chunk=False)
        elif args.backend == "gdn2-chunk":
            probe, eligibility = _gdn2(chunk=True)
        elif args.backend == "mamba3-siso":
            probe, eligibility = _mamba3(mimo=False)
        elif args.backend == "mamba3-mimo":
            probe, eligibility = _mamba3(mimo=True)
        else:
            if args.source_root is None:
                raise ValueError("flash-turing requires --source-root")
            probe, eligibility = _flash_turing(args.source_root)
        result["probe"] = probe
        result["eligibility"] = eligibility
    except Exception as error:  # noqa: BLE001 - the artifact must preserve rejection
        result["probe"] = {
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        result["eligibility"] = {
            "training_baseline_eligible": False,
            "reason": "native qualification failed; no fallback permitted",
        }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["probe"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
