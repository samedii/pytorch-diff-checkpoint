from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
from torch.nn import Module

from .first_element import first_element


class DiffCheckpoint:
    """
    A class to handle differential checkpoints for PyTorch models, saving:
    1. Parameters with requires_grad=True
    2. Parameters with different first elements
    3. New parameters not in the base model

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
        with torch.no_grad():
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            original_first_elements = {k: first_element(v).clone() for k, v in model_state_dict.items()}
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
        1. Parameters with requires_grad=True
        2. Parameters with different first elements
        3. New parameters not in the base model

        Args:
            model (Module): The PyTorch model.
            rtol (float): Relative tolerance for comparing tensor first elements.
            atol (float): Absolute tolerance for comparing tensor first elements.

        Returns:
            Dict[str, torch.Tensor]: The differential state dictionary.
        """
        with torch.no_grad():
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            model_params = {k: v.requires_grad for k, v in model.named_parameters()}
            diff_checkpoint: Dict[str, torch.Tensor] = {}

            for k, v in model_state_dict.items():
                if k not in self.original_first_elements:
                    # Save new parameters/buffers
                    diff_checkpoint[k] = v
                    continue

                current_first_element = first_element(v)
                original_first_element = self.original_first_elements[k]
                
                # Convert to same dtype for comparison
                if current_first_element.dtype != original_first_element.dtype:
                    current_first_element = current_first_element.to(original_first_element.dtype)

                # Save if:
                # 1. The tensor has changed, OR
                # 2. It's a parameter (not a buffer) and requires_grad is True
                if not torch.allclose(
                    original_first_element,
                    current_first_element,
                    rtol=rtol,
                    atol=atol,
                ) or (k in model_params and model_params[k]):
                    diff_checkpoint[k] = v

            return diff_checkpoint
