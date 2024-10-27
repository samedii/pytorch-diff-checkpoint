from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
from torch.nn import Module

from .first_element import first_element


class DiffCheckpoint:
    """
    A class to handle differential checkpoints for PyTorch models, saving modified parameters
    and all parameters with requires_grad=True.

    Attributes:
        original_first_elements (Dict[str, torch.Tensor]): A dictionary storing the original first elements of the model parameters.
    """

    original_first_elements: Dict[str, torch.Tensor]

    def __init__(self, original_first_elements: Dict[str, torch.Tensor]):
        """
        Initializes the DiffCheckpoint with its original parameter first elements.

        Args:
            original_first_elements (Dict[str, torch.Tensor]): A dictionary of original parameter first elements.
        """
        super().__init__()
        self.original_first_elements = original_first_elements

    @classmethod
    def from_base_model(cls, model: Module) -> DiffCheckpoint:
        """
        Creates a DiffCheckpoint instance from a base model by computing the first elements of its parameters.

        Args:
            model (Module): The base PyTorch model.

        Returns:
            DiffCheckpoint: An instance of DiffCheckpoint.
        """
        model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
        original_first_elements: Dict[str, torch.Tensor] = {
            k: first_element(v) for k, v in model_state_dict.items()
        }
        return cls(original_first_elements)

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

    def state_dict(
        self, model: Module, rtol: float = 1e-5, atol: float = 1e-8
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the differential state dictionary of the model, containing:
        1. Modified parameters and buffers
        2. All parameters with requires_grad=True

        Args:
            model (Module): The PyTorch model.
            rtol (float): Relative tolerance for comparing tensor first elements.
            atol (float): Absolute tolerance for comparing tensor first elements.

        Returns:
            Dict[str, torch.Tensor]: The differential state dictionary.
        """
        model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
        model_params: Dict[str, torch.Tensor] = dict(model.named_parameters())
        diff_checkpoint: Dict[str, torch.Tensor] = {}

        for k, v in model_state_dict.items():
            if k not in self.original_first_elements:
                # Save new parameters/buffers
                diff_checkpoint[k] = v
                continue

            original_first_element: torch.Tensor = self.original_first_elements[k]
            current_first_element: torch.Tensor = first_element(v)

            # Save if:
            # 1. The tensor has changed, OR
            # 2. It's a parameter (not a buffer) and requires_grad is True
            if not torch.allclose(
                original_first_element.to(current_first_element),
                current_first_element,
                rtol=rtol,
                atol=atol,
            ) or (k in model_params and model_params[k].requires_grad):
                diff_checkpoint[k] = v

        return diff_checkpoint
