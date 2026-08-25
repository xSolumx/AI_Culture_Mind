"""Execute an actual pretrained Mamba checkpoint on the local SM75 GPU.

This is deliberately fail-closed: the official ``mamba_ssm`` model class,
downloaded weights, and the checkpoint's stated tokenizer must all load.  A
randomly initialized shell, eager substitute, or different tokenizer is not a
successful probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

SEED = 20_260_825
NATURAL_TEXT = (
    "A state space model must preserve a rare fact without destroying the "
    "ordinary language structure around it. The observatory key is violet-7391. "
    "After several ordinary sentences, the observatory key remains violet-7391."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_install_provenance(source_root: Path) -> dict[str, Any]:
    """Bind the imported Mamba package to the supplied source checkout."""

    root = source_root.resolve()
    distribution = importlib.metadata.distribution("mamba-ssm")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("mamba-ssm has no direct_url.json provenance")
    direct_url = json.loads(direct_url_text)
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        raise RuntimeError("mamba-ssm is not installed from a local source tree")
    installed_from = Path(unquote(parsed.path)).resolve()
    if installed_from != root:
        raise RuntimeError(f"mamba-ssm was installed from {installed_from}, not {root}")
    return {
        "distribution": "mamba-ssm",
        "distribution_version": distribution.version,
        "direct_url": direct_url,
        "source_root": str(root),
        "source_revision": _git_revision(root),
    }


def _require_hf_revision(local_dir: Path, revision: str) -> dict[str, Any]:
    """Require Hugging Face's local-dir tree manifest for the stated commit."""

    tree_path = local_dir / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    if not tree_path.is_file():
        raise FileNotFoundError(
            f"missing Hugging Face tree manifest for revision {revision}: {tree_path}"
        )
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("format_version") != 1 or not isinstance(tree.get("files"), dict):
        raise RuntimeError(f"invalid Hugging Face tree manifest: {tree_path}")
    return {
        "revision": revision,
        "tree_manifest": str(tree_path),
        "tree_manifest_sha256": _sha256(tree_path),
    }


def _tokenizer_files(tokenizer_root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for relative in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "original/tokenizer.model",
    ):
        path = tokenizer_root / relative
        if path.is_file():
            files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if not files:
        raise FileNotFoundError(f"no tokenizer definition files in {tokenizer_root}")
    return files


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


def _require_sm75() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise RuntimeError(f"probe requires SM75, found {capability}")


def _weight_file(checkpoint: Path) -> Path:
    candidates = (
        checkpoint / "pytorch_model.bin",
        checkpoint / "model.safetensors",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no supported weight file in {checkpoint}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-source-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "checkpoint": str(args.checkpoint),
        "tokenizer_local_path": str(args.tokenizer),
        "tokenizer_source_id": args.tokenizer_source_id,
        "tokenizer_revision": args.tokenizer_revision,
        "source_root": str(args.source_root),
        "source_revision": _git_revision(args.source_root),
        "runtime": _runtime(),
        "claim_boundary": (
            "actual pretrained checkpoint execution and one fixed natural-text "
            "finite-loss probe only; differing tokenizers and pretraining corpora "
            "make the reported losses non-comparable model-quality measurements"
        ),
    }
    try:
        _require_sm75()
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        source_provenance = _source_install_provenance(args.source_root)
        model_download = _require_hf_revision(args.checkpoint, args.model_revision)
        tokenizer_download = _require_hf_revision(
            args.tokenizer, args.tokenizer_revision
        )
        weight_file = _weight_file(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )
        tokenized = tokenizer(NATURAL_TEXT, return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        if input_ids.shape[1] < 2:
            raise RuntimeError("tokenizer produced fewer than two tokens")
        dtype = torch.float16 if args.dtype == "float16" else torch.float32
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = MambaLMHeadModel.from_pretrained(
            str(args.checkpoint),
            device="cuda",
            dtype=dtype,
        )
        model.eval()
        load_seconds = time.perf_counter() - started
        torch.cuda.synchronize()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
            )
        torch.cuda.synchronize()
        next_token_id = int(logits[0, -1].argmax())
        result["probe"] = {
            "status": "pass",
            "implementation": "mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel",
            "source_provenance": source_provenance,
            "model_download": model_download,
            "tokenizer_download": tokenizer_download,
            "tokenizer_files": _tokenizer_files(args.tokenizer),
            "checkpoint_weight_file": weight_file.name,
            "checkpoint_weight_bytes": weight_file.stat().st_size,
            "checkpoint_weight_sha256": _sha256(weight_file),
            "config_sha256": _sha256(args.checkpoint / "config.json"),
            "dtype": str(dtype),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "tokenizer_class": type(tokenizer).__name__,
            "token_count": int(input_ids.shape[1]),
            "token_ids_sha256": hashlib.sha256(
                input_ids.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "logits_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "finite_loss": bool(torch.isfinite(loss)),
            "natural_text_loss": float(loss),
            "next_token_id": next_token_id,
            "next_token_text": tokenizer.decode([next_token_id]),
            "load_seconds": load_seconds,
            "forward_seconds_first_call": time.perf_counter() - forward_started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        if not result["probe"]["finite_logits"] or not result["probe"]["finite_loss"]:
            raise RuntimeError("pretrained checkpoint produced non-finite output")
        result["eligibility"] = {
            "actual_pretrained_checkpoint_executed": True,
            "quality_baseline_eligible": False,
            "reason": (
                "checkpoint and declared tokenizer executed natively on SM75; "
                "this single unmatched text is not a quality benchmark"
            ),
        }
    except Exception as error:  # noqa: BLE001 - rejection must be preserved
        result["probe"] = {
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        result["eligibility"] = {
            "actual_pretrained_checkpoint_executed": False,
            "quality_baseline_eligible": False,
            "reason": "native checkpoint qualification failed; no fallback permitted",
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["probe"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
