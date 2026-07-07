"""Automatic device discovery and allocation.

The default policy is intentionally simple:

- 0 CUDA GPUs: classifier and optimizer LLM run on CPU.
- 1 CUDA GPU: classifier uses cuda:0; optimizer LLM runs on CPU.
- 2 CUDA GPUs: classifier uses cuda:0; one optimizer LLM uses cuda:1.
- 3 CUDA GPUs: classifier uses cuda:0; two optimizer LLMs use cuda:1-2.
- 4+ CUDA GPUs: classifier uses cuda:0; three optimizer LLMs use cuda:1-3.

The paper still uses three logical LLM generations per optimization step. When
fewer than three physical optimizer models are available, those generations are
scheduled sequentially across the available model instance(s).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch


def _split_devices(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalize_device(value: str, cuda_count: int) -> str:
    raw = str(value).strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        raw = "cuda:0"
    elif raw.isdigit():
        raw = f"cuda:{raw}"

    if raw.startswith("cuda:"):
        try:
            index = int(raw.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {value!r}") from exc
        if index < 0 or index >= cuda_count:
            raise ValueError(
                f"Requested {raw}, but PyTorch can see only {cuda_count} CUDA device(s)."
            )
        return f"cuda:{index}"

    raise ValueError(f"Unsupported device {value!r}; use cpu or cuda:N")


def _unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def normalize_devices(values: Iterable[str]) -> list[str]:
    """Normalize and validate an explicit list of CPU/CUDA devices."""
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return _unique(_normalize_device(value, cuda_count) for value in values)


@dataclass(frozen=True)
class HardwarePlan:
    cuda_count: int
    training_devices: tuple[str, ...]
    classifier_device: str
    optimizer_devices: tuple[str, ...]
    logical_optimizer_copies: int

    @property
    def physical_optimizer_copies(self) -> int:
        return len(self.optimizer_devices)

    def describe(self) -> str:
        gpu_word = "GPU" if self.cuda_count == 1 else "GPUs"
        return (
            f"Detected {self.cuda_count} CUDA {gpu_word}. "
            f"Training devices: {', '.join(self.training_devices)}. "
            f"Classifier device: {self.classifier_device}. "
            f"Optimizer model device(s): {', '.join(self.optimizer_devices)}. "
            f"Logical optimizer generations per step: {self.logical_optimizer_copies}."
        )


def build_hardware_plan(logical_optimizer_copies: int = 3) -> HardwarePlan:
    if logical_optimizer_copies < 1:
        raise ValueError("logical_optimizer_copies must be at least 1")

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    visible_cuda = [f"cuda:{index}" for index in range(cuda_count)]

    training_override = _split_devices(os.getenv("PAPER_DEVICES"))
    if training_override:
        training_devices = _unique(
            _normalize_device(device, cuda_count) for device in training_override
        )
    else:
        training_devices = visible_cuda or ["cpu"]

    classifier_override = os.getenv("CLASSIFIER_DEVICE")
    if classifier_override:
        classifier_device = _normalize_device(classifier_override, cuda_count)
    else:
        classifier_device = training_devices[0]

    optimizer_override = _split_devices(os.getenv("OPTIMIZER_DEVICES"))
    if not optimizer_override:
        # Backward-compatible alias used by earlier repository versions.
        optimizer_override = _split_devices(os.getenv("OPTIMIZER_GPU_IDS"))

    if optimizer_override:
        optimizer_devices = _unique(
            _normalize_device(device, cuda_count) for device in optimizer_override
        )
    else:
        optimizer_devices = [
            device for device in visible_cuda if device != classifier_device
        ][:logical_optimizer_copies]
        if not optimizer_devices:
            optimizer_devices = ["cpu"]

    return HardwarePlan(
        cuda_count=cuda_count,
        training_devices=tuple(training_devices),
        classifier_device=classifier_device,
        optimizer_devices=tuple(optimizer_devices),
        logical_optimizer_copies=logical_optimizer_copies,
    )


def pipeline_device(device: str) -> int:
    """Return the Transformers pipeline device argument for a normalized device."""
    if device == "cpu":
        return -1
    return int(device.split(":", 1)[1])


def optimizer_dtype(device: str) -> torch.dtype:
    """Choose a broadly supported inference dtype for the optimizer model."""
    if device == "cpu":
        return torch.float32

    index = int(device.split(":", 1)[1])
    try:
        with torch.cuda.device(index):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
    except (AttributeError, RuntimeError):
        pass
    return torch.float16
