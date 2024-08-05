import tempfile
from pathlib import Path

import pytest
import torch
from torch.nn import Module

from diff_checkpoint import DiffCheckpoint


class SimpleModel(Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def setup_model():
    model = SimpleModel()
    diff_checkpoint_path = Path("test_diff_checkpoint.pth")
    model_state_dict = model.state_dict()
    yield model, diff_checkpoint_path, model_state_dict

    if diff_checkpoint_path.exists():
        diff_checkpoint_path.unlink()


def test_modify_weights_and_save(setup_model):
    """
    This test checks that modified weight parameters are saved in the differential
    checkpoint, while non-weight parameters are not included.
    """
    model, diff_checkpoint_path, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.add_(0.1)

    diff_checkpoint.save(diff_checkpoint_path)

    saved_diff_checkpoint = torch.load(diff_checkpoint_path)
    for k, v in model_state_dict.items():
        if "weight" in k:
            assert (
                k in saved_diff_checkpoint
            ), f"Weight parameter '{k}' not found in saved diff checkpoint"
        else:
            assert (
                k not in saved_diff_checkpoint
            ), f"Non-weight parameter '{k}' should not be in saved diff checkpoint"


def test_load_diff_checkpoint(setup_model):
    """
    This test ensures that modified weight parameters are correctly loaded from
    the differential checkpoint back into the model.
    """
    model, diff_checkpoint_path, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.add_(0.1)

    diff_checkpoint.save(diff_checkpoint_path)

    # Load the differential checkpoint back into the model
    loaded_diff_checkpoint = torch.load(diff_checkpoint_path)
    model.load_state_dict(loaded_diff_checkpoint, strict=False)

    for name, param in model.named_parameters():
        if "weight" in name:
            assert torch.allclose(
                param, model.state_dict()[name]
            ), f"Weight parameter '{name}' not correctly loaded from diff checkpoint"


def test_no_changes_after_save_and_load(setup_model):
    """
    This test verifies that no weights are saved in the differential checkpoint
    if no modifications have been made to the model.
    """
    model, diff_checkpoint_path, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    diff_checkpoint.save(diff_checkpoint_path)
    loaded_diff_checkpoint = torch.load(diff_checkpoint_path)

    # Ensure no weights are saved since they have not been modified
    assert len(loaded_diff_checkpoint) == 0


def test_partial_weight_modification(setup_model):
    """
    This test checks that only the modified subset of weight parameters are saved
    in the differential checkpoint.
    """
    model, diff_checkpoint_path, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify only a subset of the weights
    modified_weights = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "fc1" in name:
                param.add_(0.1)
                modified_weights.append(name)

    diff_checkpoint.save(diff_checkpoint_path)
    saved_diff_checkpoint = torch.load(diff_checkpoint_path)

    for name, param in model_state_dict.items():
        if name in modified_weights:
            assert (
                name in saved_diff_checkpoint
            ), f"Modified weight '{name}' not found in saved diff checkpoint"
        else:
            assert (
                name not in saved_diff_checkpoint
            ), f"Unmodified weight '{name}' should not be in saved diff checkpoint"


def test_non_weight_parameter_modification(setup_model):
    """
    This test ensures that modifications to non-weight parameters (e.g., biases)
    are not saved in the differential checkpoint.
    """
    model, diff_checkpoint_path, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify non-weight parameters (e.g., biases)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name:
                param.add_(0.1)

    diff_checkpoint.save(diff_checkpoint_path)
    saved_diff_checkpoint = torch.load(diff_checkpoint_path)

    for name, param in model_state_dict.items():
        if "weight" in name:
            assert (
                name not in saved_diff_checkpoint
            ), f"Non-weight parameter '{name}' should not be in saved diff checkpoint"


def test_save_and_load_with_different_models(setup_model):
    """
    This test verifies that a differential checkpoint saved from one model can be
    correctly loaded into a new model of the same architecture.
    """
    model, diff_checkpoint_path, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.add_(0.1)

    diff_checkpoint.save(diff_checkpoint_path)

    # Create a new model of the same architecture
    new_model = type(model)()
    new_diff_checkpoint = DiffCheckpoint.from_base_model(new_model)

    # Load the differential checkpoint into the new model
    loaded_diff_checkpoint = torch.load(diff_checkpoint_path)
    new_model.load_state_dict(loaded_diff_checkpoint, strict=False)

    for name, param in new_model.named_parameters():
        if "weight" in name:
            assert torch.allclose(
                param, model.state_dict()[name]
            ), f"Weight parameter '{name}' not correctly loaded into new model from diff checkpoint"
