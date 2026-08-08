"""Matched patch-sequence image benchmarks for Spinor, Mamba-2, and Mamba-3.

The benchmark deliberately uses continuous patch embeddings for every model.
This avoids giving Mamba a tokenization shortcut and makes the comparison a
sequence-model comparison rather than a language-model API comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import Mamba2Config, Mamba2Model
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from mamba3_reference import Mamba3Mixer, RMSNorm
from spinor_delta_ssm import GeometricRMSNorm, SpinorDeltaBlock


@dataclass
class Config:
    steps: int = 100
    batch_size: int = 32
    eval_batch_size: int = 64
    patch_size: int = 4
    # The vision head has a different parameter mix from the byte LM.  Forty-
    # four Cl(3) channels puts the complete local model near the Mamba baselines
    # after the patch and classifier heads are included.
    channels: int = 44
    layers: int = 4
    expansion: int = 2
    mamba2_width: int = 160
    mamba3_width: int = 152
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_samples: int = 0
    eval_samples: int = 0
    eval_batches: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != images.shape[-2]:
        raise ValueError("images must have shape (batch, channels, square, square)")
    size = images.shape[-1]
    if size % patch_size:
        raise ValueError("image size must be divisible by patch_size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # B, C, rows, cols, patch_h, patch_w -> B, tokens, C*patch_h*patch_w.
    return patches.permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2)


class SpinorVisionClassifier(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        classes: int,
        channels: int,
        layers: int,
        expansion: int,
        sequence_length: int,
    ):
        super().__init__()
        self.channels = channels
        self.patch_projection = nn.Linear(patch_dim, channels * 8)
        self.class_token = nn.Parameter(torch.zeros(1, 1, channels, 8))
        self.position = nn.Parameter(torch.zeros(1, sequence_length, channels, 8))
        self.blocks = nn.ModuleList(
            [SpinorDeltaBlock(channels, expansion) for _ in range(layers)]
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.classifier = nn.Linear(channels * 8, classes)

    def forward(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        patches = patchify(images, patch_size)
        outputs = self.patch_projection(patches).reshape(
            images.shape[0], patches.shape[1], self.channels, 8
        )
        cls = self.class_token.expand(images.shape[0], -1, -1, -1)
        outputs = torch.cat((cls, outputs), dim=1) + self.position
        for block in self.blocks:
            outputs, _ = block(outputs)
        outputs = self.final_norm(outputs[:, 0])
        return self.classifier(outputs.flatten(-2))


class Mamba2VisionClassifier(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        classes: int,
        width: int,
        layers: int,
        sequence_length: int,
    ):
        super().__init__()
        self.patch_projection = nn.Linear(patch_dim, width)
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        self.backbone = Mamba2Model(
            Mamba2Config(
                vocab_size=1,
                hidden_size=width,
                state_size=16,
                num_hidden_layers=layers,
                num_heads=5,
                head_dim=64,
                expand=2,
                conv_kernel=4,
                n_groups=1,
                use_cache=False,
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
            )
        )
        # Continuous inputs bypass the vocabulary embedding. Remove that
        # unused parameter so optimizer state, gradients, and parameter counts
        # describe the actual image model rather than a dead LM table.
        self.backbone.embeddings = nn.Identity()
        self.classifier = nn.Linear(width, classes)

    def forward(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        patches = self.patch_projection(patchify(images, patch_size))
        cls = self.class_token.expand(images.shape[0], -1, -1)
        outputs = torch.cat((cls, patches), dim=1) + self.position
        outputs = self.backbone(
            inputs_embeds=outputs, return_dict=True
        ).last_hidden_state
        return self.classifier(outputs[:, 0])


class Mamba3VisionClassifier(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        classes: int,
        width: int,
        layers: int,
        sequence_length: int,
    ):
        super().__init__()
        if width % 4:
            raise ValueError("Mamba-3 width must be divisible by headdim")
        self.patch_projection = nn.Linear(patch_dim, width)
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": RMSNorm(width),
                        "mixer": Mamba3Mixer(
                            width,
                            d_state=16,
                            headdim=width // 4,
                            mimo_rank=2,
                        ),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        patches = self.patch_projection(patchify(images, patch_size))
        cls = self.class_token.expand(images.shape[0], -1, -1)
        outputs = torch.cat((cls, patches), dim=1) + self.position
        for layer in self.layers:
            outputs = outputs + layer["mixer"](layer["norm"](outputs))
        return self.classifier(self.final_norm(outputs[:, 0]))


class HFParquetImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, label_column: str):
        self.rows = load_dataset("parquet", data_files=str(path), split="train")
        self.label_column = label_column
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = row["img"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(row[self.label_column])


def hf_parquet_dataset(repo: str, path: str, label_column: str, root: Path):
    local = hf_hub_download(
        repo,
        filename=path,
        repo_type="dataset",
        local_dir=str(root / "hf"),
    )
    return HFParquetImageDataset(Path(local), label_column)


def dataset_spec(name: str, root: Path):
    name = name.lower()
    common = [transforms.ToTensor()]
    if name == "cifar100":
        dataset = hf_parquet_dataset(
            "uoft-cs/cifar100",
            "cifar100/train-00000-of-00001.parquet",
            "fine_label",
            root,
        )
        test = hf_parquet_dataset(
            "uoft-cs/cifar100",
            "cifar100/test-00000-of-00001.parquet",
            "fine_label",
            root,
        )
        return dataset, test, 3, 100
    if name == "cifar10":
        dataset = hf_parquet_dataset(
            "uoft-cs/cifar10",
            "plain_text/train-00000-of-00001.parquet",
            "label",
            root,
        )
        test = hf_parquet_dataset(
            "uoft-cs/cifar10",
            "plain_text/test-00000-of-00001.parquet",
            "label",
            root,
        )
        return dataset, test, 3, 10
    if name == "mnist":
        common = [transforms.Pad(2), transforms.ToTensor()]
        dataset = datasets.MNIST(
            root=root, train=True, download=True, transform=transforms.Compose(common)
        )
        test = datasets.MNIST(
            root=root, train=False, download=True, transform=transforms.Compose(common)
        )
        return dataset, test, 1, 10
    raise ValueError(f"unknown dataset: {name}")


def limited(dataset, count: int, seed: int):
    if not count or count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(
    name: str, patch_dim: int, classes: int, config: Config, sequence_length: int
):
    if name == "spinor":
        return SpinorVisionClassifier(
            patch_dim,
            classes,
            config.channels,
            config.layers,
            config.expansion,
            sequence_length,
        )
    if name == "mamba2":
        return Mamba2VisionClassifier(
            patch_dim, classes, config.mamba2_width, config.layers, sequence_length
        )
    if name == "mamba3":
        return Mamba3VisionClassifier(
            patch_dim, classes, config.mamba3_width, config.layers, sequence_length
        )
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def evaluate(
    model, loader: Iterable, device: torch.device, patch_size: int, max_batches: int
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for index, (images, labels) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        logits = model(images, patch_size)
        total_loss += (
            float(nn.functional.cross_entropy(logits, labels)) * labels.numel()
        )
        total_correct += int((logits.argmax(dim=-1) == labels).sum())
        total += labels.numel()
    if total == 0:
        raise ValueError("evaluation loader produced no examples")
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "examples": total,
    }


def train_one(
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    patch_size: int,
    config: Config,
    classes: int,
    device: torch.device,
    seed: int,
):
    seed_everything(seed)
    # The model is constructed only after seeding, matching the corrected
    # language harness and preventing initialization leakage between models.
    patch_dim = int(train_loader.dataset[0][0].shape[0] * patch_size * patch_size)
    sequence_length = (train_loader.dataset[0][0].shape[-1] // patch_size) ** 2 + 1
    model = build_model(model_name, patch_dim, classes, config, sequence_length).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial = evaluate(model, test_loader, device, patch_size, config.eval_batches)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    iterator = iter(train_loader)
    losses = []
    start = time.perf_counter()
    seen = 0
    for step in range(config.steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, patch_size)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        seen += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final = evaluate(model, test_loader, device, patch_size, config.eval_batches)
    return {
        "name": model_name,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_loss": initial["loss"],
        "final_loss": final["loss"],
        "accuracy": final["accuracy"],
        "evaluated_examples": final["examples"],
        "final_train_loss": losses[-1],
        "mean_last_20_train_loss": float(np.mean(losses[-20:])),
        "elapsed_seconds": elapsed,
        "examples_per_second": seen / elapsed,
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
    }


def digest_dataset(dataset) -> str:
    digest = hashlib.sha256()
    # Hash raw tensor bytes and labels from the deterministic dataset view.
    for image, label in dataset:
        digest.update(image.contiguous().numpy().tobytes())
        digest.update(int(label).to_bytes(2, "little", signed=False))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cifar100,cifar10,mnist")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=0)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    device = torch.device(args.device)
    config = Config(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        patch_size=args.patch_size,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        eval_batches=args.eval_batches,
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    report = {
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "config": asdict(config),
        "datasets": {},
        "integrity": {
            "model_initialized_after_seed": True,
            "python_numpy_torch_cuda_seeded": True,
            "continuous_patch_protocol": True,
            "mamba2_backend": "transformers_mamba2_model",
            "mamba3_backend": "pure_pytorch_reference",
            "spinor_backend": "tensor_cuda_associative_scan",
        },
    }
    for dataset_name in [
        x.strip().lower() for x in args.datasets.split(",") if x.strip()
    ]:
        train, test, channels, classes = dataset_spec(dataset_name, args.data_root)
        # Fixed subsets make quick screens reproducible and keep full-data runs
        # available by passing zero (the default).
        train = limited(train, config.train_samples, 1729)
        test = limited(test, config.eval_samples, 1730)
        image_size = train[0][0].shape[-1]
        patch_dim = channels * config.patch_size * config.patch_size
        sequence_length = (image_size // config.patch_size) ** 2 + 1
        dataset_report = {
            "classes": classes,
            "train_examples": len(train),
            "test_examples": len(test),
            "image_size": image_size,
            "channels": channels,
            "patch_size": config.patch_size,
            "sequence_length": sequence_length,
            "train_digest": digest_dataset(train),
            "test_digest": digest_dataset(test),
            "results": [],
        }
        for seed in seeds:
            for model_name in ("spinor", "mamba2", "mamba3"):
                # Recreate both loaders for every run so every model sees the
                # identical deterministic batch sequence rather than inheriting
                # the previous model's shuffled-loader state.
                train_loader = make_loader(train, config.batch_size, True, 2000)
                test_loader = make_loader(test, config.eval_batch_size, False, 2001)
                result = train_one(
                    model_name,
                    train_loader,
                    test_loader,
                    config.patch_size,
                    config,
                    classes,
                    device,
                    seed,
                )
                dataset_report["results"].append(result)
                print(
                    f"{dataset_name} {model_name} seed={seed} "
                    f"acc={result['accuracy']:.4f} loss={result['final_loss']:.4f}",
                    flush=True,
                )
        report["datasets"][dataset_name] = dataset_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
