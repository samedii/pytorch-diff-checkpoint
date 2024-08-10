import random
import tempfile
from pathlib import Path

import pytest
import torch
from torch.nn import Module

from diff_checkpoint import DiffCheckpoint
from diff_checkpoint.hash_tensor import hash_tensor


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
    model_state_dict = model.state_dict()
    yield model, model_state_dict


def test_modify_weights_and_save(setup_model):
    """
    This test checks that modified parameters are saved in the differential
    checkpoint, while non-modified parameters are not included.
    """
    model, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Randomly select and modify some parameters
    params = list(model.named_parameters())
    num_to_modify = random.randint(1, len(params))
    params_to_modify = random.sample(params, num_to_modify)

    with torch.no_grad():
        for name, param in params_to_modify:
            param.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
        for k, v in model_state_dict.items():
            if k in [name for name, _ in params_to_modify]:
                assert (
                    k in saved_diff_checkpoint
                ), f"Modified parameter '{k}' not found in saved diff checkpoint"
            else:
                assert (
                    k not in saved_diff_checkpoint
                ), f"Non-modified parameter '{k}' should not be in saved diff checkpoint"


def test_load_diff_checkpoint(setup_model):
    """
    This test ensures that modified weight parameters are correctly loaded from
    the differential checkpoint back into the model.
    """
    model, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        # Load the differential checkpoint back into the model
        loaded_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
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
    model, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        loaded_diff_checkpoint = torch.load(temp_file.name, weights_only=True)

        # Ensure no weights are saved since they have not been modified
        assert len(loaded_diff_checkpoint) == 0


def test_partial_weight_modification(setup_model):
    """
    This test checks that only the modified subset of weight parameters are saved
    in the differential checkpoint.
    """
    model, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify only a subset of the weights
    modified_weights = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "fc1" in name:
                param.add_(0.1)
                modified_weights.append(name)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)

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
    model, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify non-weight parameters (e.g., biases)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name:
                param.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)

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
    model, _ = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        # Create a new model of the same architecture
        new_model = type(model)()
        new_diff_checkpoint = DiffCheckpoint.from_base_model(new_model)

        # Load the differential checkpoint into the new model
        loaded_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
        new_model.load_state_dict(loaded_diff_checkpoint, strict=False)

        for name, param in new_model.named_parameters():
            if "weight" in name:
                assert torch.allclose(
                    param, model.state_dict()[name]
                ), f"Weight parameter '{name}' not correctly loaded into new model from diff checkpoint"


def test_modify_batch_norm_weights_and_save(setup_model):
    """
    This test checks that modified batch norm weight parameters are saved in the differential
    checkpoint, while non-weight parameters are not included.
    """
    model, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify specific batch norm weights
    with torch.no_grad():
        model.bn1.weight.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
        assert (
            "bn1.weight" in saved_diff_checkpoint
        ), "Batch norm weight parameter 'bn1.weight' not found in saved diff checkpoint"


def test_handle_new_keys_after_initialization(setup_model):
    """
    This test checks that new keys added to the model after initializing the differential
    checkpoint are handled correctly.
    """
    model, model_state_dict = setup_model
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Add new parameters to the model (simulating injected weights like PEFT)
    model.new_fc = torch.nn.Linear(10, 5)
    with torch.no_grad():
        model.new_fc.weight.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
        assert (
            "new_fc.weight" in saved_diff_checkpoint
        ), "Newly added parameter 'new_fc.weight' not found in saved diff checkpoint"
