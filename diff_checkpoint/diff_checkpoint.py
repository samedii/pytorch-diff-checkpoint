from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Union

import torch
from torch.nn import Module


class DiffCheckpoint:
    model: Module
    original_hashes: Dict[str, str]

    def __init__(self, model: Module, original_hashes: Dict[str, str] = None):
        super().__init__()
        self.model = model
        self.original_hashes = original_hashes or {}

    @classmethod
    def from_base_model(cls, model: Module) -> DiffCheckpoint:
        model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
        original_hashes: Dict[str, str] = {
            k: cls._hash_tensor(v) for k, v in model_state_dict.items() if "weight" in k
        }
        return cls(model, original_hashes)

    @staticmethod
    def _hash_tensor(tensor: torch.Tensor) -> str:
        if tensor.numel() == 0:
            return hashlib.sha256(b"").hexdigest()
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    def save(self, path: Union[str, Path]) -> DiffCheckpoint:
        model_state_dict: Dict[str, torch.Tensor] = self.model.state_dict()
        model_buffers: Dict[str, torch.Tensor] = {
            k: v for k, v in self.model.named_buffers()
        }
        diff_checkpoint: Dict[str, torch.Tensor] = {}

        for k, v in {**model_state_dict, **model_buffers}.items():
            if "weight" in k:  # Only consider weight parameters
                original_hash: str = self.original_hashes.get(k, "")
                current_hash: str = self._hash_tensor(v)
                if original_hash != current_hash:
                    diff_checkpoint[k] = v

        torch.save(diff_checkpoint, str(path))
        return self
