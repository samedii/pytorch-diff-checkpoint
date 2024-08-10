from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
from torch.nn import Module

from .hash_tensor import hash_tensor


class DiffCheckpoint:
    """
    A class to handle differential checkpoints for PyTorch models, saving only the modified parameters.

    Attributes:
        original_hashes (Dict[str, str]): A dictionary storing the original hashes of the model parameters.
    """

    original_hashes: Dict[str, str]

    def __init__(self, original_hashes: Dict[str, str]):
        """
        Initializes the DiffCheckpoint with its original parameter hashes.

        Args:
            original_hashes (Dict[str, str]): A dictionary of original parameter hashes.
        """
        super().__init__()
        self.original_hashes = original_hashes

    @classmethod
    def from_base_model(cls, model: Module) -> DiffCheckpoint:
        """
        Creates a DiffCheckpoint instance from a base model by computing the hashes of its parameters.

        Args:
            model (Module): The base PyTorch model.

        Returns:
            DiffCheckpoint: An instance of DiffCheckpoint.
        """
        model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
        original_hashes: Dict[str, str] = {
            k: hash_tensor(v) for k, v in model_state_dict.items()
        }
        return cls(original_hashes)

    def save(self, model: Module, path: Union[str, Path]) -> DiffCheckpoint:
        """
        Saves the differential checkpoint to a file.

        Args:
            model (Module): The PyTorch model.
            path (Union[str, Path]): The file path to save the checkpoint.

        Returns:
            DiffCheckpoint: The current instance of DiffCheckpoint.
        """
        diff_checkpoint = self.state_dict(model)
        torch.save(diff_checkpoint, str(path))
        return self

    def state_dict(self, model: Module) -> Dict[str, torch.Tensor]:
        """
        Returns the differential state dictionary of the model, containing only modified parameters.

        Args:
            model (Module): The PyTorch model.

        Returns:
            Dict[str, torch.Tensor]: The differential state dictionary.
        """
        model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
        model_buffers: Dict[str, torch.Tensor] = {
            k: v for k, v in model.named_buffers()
        }
        model_params: Dict[str, torch.Tensor] = {
            k: v for k, v in model.named_parameters()
        }
        diff_checkpoint: Dict[str, torch.Tensor] = {}

        for k, v in {**model_state_dict, **model_buffers, **model_params}.items():
            original_hash: str = self.original_hashes.get(k, "")
            current_hash: str = hash_tensor(v)
            if original_hash != current_hash:
                diff_checkpoint[k] = v

        return diff_checkpoint
