"""Backbone and prototype-network definitions used in the paper."""
from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.config import CLASSIFIER_DEVICE, MODEL_IDS, PROTOTYPE_DIM


def _iter_module_list(value: object) -> list[nn.Module] | None:
    if isinstance(value, (nn.ModuleList, list, tuple)) and len(value) > 0:
        return list(value)
    return None


def _find_final_transformer_block(model: nn.Module) -> nn.Module:
    """Locate the final encoder block across the five supported architectures."""
    candidate_paths = (
        ("encoder", "layer"),
        ("electra", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("mpnet", "encoder", "layer"),
        ("layers",),  # ModernBERT
        ("encoder", "layers"),
    )
    for path in candidate_paths:
        obj: object = model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        blocks = _iter_module_list(obj)
        if blocks:
            return blocks[-1]
    raise RuntimeError(
        f"Could not locate the final transformer block for {model.__class__.__name__}. "
        "Update _find_final_transformer_block for the installed Transformers version."
    )


class ModelWrapper(nn.Module):
    """A pretrained text encoder with only its final block and 256-D projection trainable."""

    def __init__(
        self,
        model_name: str,
        device: str = CLASSIFIER_DEVICE,
        prototype_dim: int = PROTOTYPE_DIM,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(MODEL_IDS)}")

        self.model_name = model_name
        self.model_id = MODEL_IDS[model_name]
        self.device_name = device
        token = hf_token or os.getenv("HF_TOKEN")

        self.config = AutoConfig.from_pretrained(self.model_id, token=token)
        self.hugging_model = AutoModel.from_pretrained(self.model_id, token=token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token, use_fast=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        for parameter in self.hugging_model.parameters():
            parameter.requires_grad = False
        for parameter in _find_final_transformer_block(self.hugging_model).parameters():
            parameter.requires_grad = True

        hidden_size = int(getattr(self.config, "hidden_size"))
        self.projection = nn.Linear(hidden_size, prototype_dim)
        self.prototype_dim = prototype_dim
        self.model_type = "encoder"
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.hugging_model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            return_dict=True,
        )
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_representation)


class LMProtoNet(nn.Module):
    """Prototype classifier using cosine similarities as activations."""

    def __init__(
        self,
        backbone: ModelWrapper,
        num_labels: int,
        num_protos_per_class: int,
        init_prototypes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_labels = int(num_labels)
        self.num_protos_per_class = int(num_protos_per_class)
        self.num_total_prototypes = self.num_labels * self.num_protos_per_class

        expected = (self.num_total_prototypes, self.backbone.prototype_dim)
        if init_prototypes is None:
            prototype_tensor = torch.empty(expected, device=self.backbone.device)
            nn.init.xavier_uniform_(prototype_tensor)
        else:
            if tuple(init_prototypes.shape) != expected:
                raise ValueError(f"Prototype shape {tuple(init_prototypes.shape)} != expected {expected}")
            prototype_tensor = init_prototypes.to(self.backbone.device).clone()

        self.prototypes = nn.Parameter(prototype_tensor)
        self.classifier = nn.Linear(self.num_total_prototypes, self.num_labels, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.to(self.backbone.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        representation = F.normalize(representation, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        similarities = representation @ prototypes.T

        # Equation (5): prototype-to-nearest-minibatch-example loss.
        interpretability_loss = (1.0 - similarities).min(dim=0).values.mean()
        # Equation (6): example-to-nearest-prototype clustering loss.
        clustering_loss = (1.0 - similarities).min(dim=1).values.mean()
        # Equation (7): unique prototype-pair separation loss.
        prototype_similarity = prototypes @ prototypes.T
        mask = torch.triu(torch.ones_like(prototype_similarity, dtype=torch.bool), diagonal=1)
        separation_loss = (1.0 + prototype_similarity[mask]).mean()

        logits = self.classifier(similarities)
        return {
            "logits": logits,
            "acts": similarities,
            "representation": representation,
            "l_interpretability": interpretability_loss,
            "l_clustering": clustering_loss,
            "l_separation": separation_loss,
        }
