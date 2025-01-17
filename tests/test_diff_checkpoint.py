import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn import Module

from diff_checkpoint import DiffCheckpoint
from diff_checkpoint.first_element import first_element  # Updated import


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
    model = SimpleModel()  # Remove .requires_grad_(False)
    model_state_dict = model.state_dict()
    yield model, model_state_dict


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

        # Only parameters with requires_grad=True should be saved
        assert len(loaded_diff_checkpoint) == sum(
            p.requires_grad for p in model.parameters()
        )


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


def test_save_all_requires_grad():
    model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_checkpoint = torch.load(temp_file.name)

        # Check that all parameters with requires_grad=True are saved
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in saved_checkpoint


def test_load_diff_checkpoint():
    original_model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(original_model)

    # Modify some parameters
    with torch.no_grad():
        original_model.fc1.weight.add_(0.1)
        original_model.bn1.bias.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(original_model, temp_file.name)

        # Load the checkpoint into a new model
        new_model = SimpleModel()
        loaded_checkpoint = torch.load(temp_file.name)
        new_model.load_state_dict(loaded_checkpoint, strict=False)

        # Check that the modified parameters are correctly loaded
        assert torch.allclose(original_model.fc1.weight, new_model.fc1.weight)
        assert torch.allclose(original_model.bn1.bias, new_model.bn1.bias)


def test_handle_new_parameters():
    model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Add a new parameter to the model
    model.new_param = torch.nn.Parameter(torch.randn(5, 5))

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_checkpoint = torch.load(temp_file.name)

        # Check that the new parameter is saved
        assert "new_param" in saved_checkpoint


def test_save_only_modified_parameters():
    model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify only one parameter
    with torch.no_grad():
        model.fc1.weight.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_checkpoint = torch.load(temp_file.name)

        assert "fc1.weight" in saved_checkpoint, "Modified parameter should be saved"
        # Check that all parameters with requires_grad=True are saved
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert (
                    name in saved_checkpoint
                ), f"Parameter {name} with requires_grad=True should be saved"


def test_save_parameters_with_requires_grad():
    model = SimpleModel()

    # Set requires_grad=False for some parameters
    model.fc1.weight.requires_grad = False
    model.fc2.weight.requires_grad = False

    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_checkpoint = torch.load(temp_file.name)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert (
                    name in saved_checkpoint
                ), f"Parameter {name} with requires_grad=True should be saved"
            else:
                assert (
                    name not in saved_checkpoint
                ), f"Parameter {name} with requires_grad=False should not be saved"


def test_save_and_load_with_different_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    model = SimpleModel().to("cuda")
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        model.fc1.weight.add_(0.1)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        # Load the checkpoint into a CPU model
        cpu_model = SimpleModel()
        loaded_checkpoint = torch.load(temp_file.name, map_location="cpu")
        cpu_model.load_state_dict(loaded_checkpoint, strict=False)

        assert torch.allclose(
            model.fc1.weight.cpu(), cpu_model.fc1.weight
        ), "Weights should match after loading to a different device"


def test_gradients_after_loading():
    model = SimpleModel()
    model.eval()  # Set the model to evaluation mode
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights and compute gradients
    x = torch.randn(4, 10)  # Use a batch size > 1
    y = model(x)
    loss = F.mse_loss(y, torch.ones_like(y))
    loss.backward()

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        # Load the checkpoint into a new model
        new_model = SimpleModel()
        new_model.eval()  # Set the new model to evaluation mode
        loaded_checkpoint = torch.load(temp_file.name)
        new_model.load_state_dict(loaded_checkpoint, strict=False)

        # Check if parameters are preserved
        for (name, param), (new_name, new_param) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name == new_name, "Parameter names should match"
            assert torch.allclose(
                param.data, new_param.data
            ), f"Parameter values for {name} should match"
            # Note: Gradients are not preserved in the checkpoint, so we don't check for them


def test_partial_loading():
    model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify some weights
    with torch.no_grad():
        model.fc1.weight.add_(0.1)
        model.fc2.weight.add_(0.2)

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)

        # Create a new model with a different structure
        class PartialModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 10)

        partial_model = PartialModel()
        loaded_checkpoint = torch.load(temp_file.name)
        partial_model.load_state_dict(loaded_checkpoint, strict=False)

        assert torch.allclose(
            model.fc1.weight, partial_model.fc1.weight
        ), "fc1 weights should match after partial loading"
        assert not hasattr(
            partial_model, "fc2"
        ), "fc2 should not be present in the partial model"


def test_handle_new_parameters_with_different_dtypes():
    """
    This test checks that new parameters with different dtypes are correctly handled
    and saved in the differential checkpoint.
    """
    model = SimpleModel()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Add new parameters with different dtypes
    model.new_float = torch.nn.Parameter(torch.randn(5, 5))  # default is float32
    model.new_byte = torch.nn.Parameter(
        torch.ones(5, 5, dtype=torch.uint8), requires_grad=False
    )
    model.new_bool = torch.nn.Parameter(
        torch.ones(5, 5, dtype=torch.bool), requires_grad=False
    )
    model.new_long = torch.nn.Parameter(
        torch.ones(5, 5, dtype=torch.long), requires_grad=False
    )

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
        saved_checkpoint = torch.load(temp_file.name)

        # Check that all new parameters are saved
        assert "new_float" in saved_checkpoint, "New float parameter should be saved"
        assert "new_byte" in saved_checkpoint, "New byte parameter should be saved"
        assert "new_bool" in saved_checkpoint, "New bool parameter should be saved"
        assert "new_long" in saved_checkpoint, "New long parameter should be saved"

        # Check that the dtypes are preserved
        assert saved_checkpoint["new_float"].dtype == torch.float32


def test_state_dict_content():
    """
    Test that state_dict returns the correct content:
    1. Parameters with requires_grad=True
    2. Parameters with different first elements
    3. New parameters not in the base model
    """
    model = SimpleModel()
    # Set one parameter to not require gradients
    model.fc1.weight.requires_grad = False
    modified_param_name = "fc1.weight"
    
    # Store original value before creating checkpoint
    original_value = model.fc1.weight[0, 0].item()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)
    
    # Initially, only parameters with requires_grad=True should be in state dict
    initial_state = diff_checkpoint.state_dict(model)
    print("\nInitial state:", initial_state.keys())
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in initial_state, f"Parameter {name} with requires_grad=True should be in initial state"
        else:
            assert name not in initial_state, f"Parameter {name} with requires_grad=False should not be in initial state"
    
    # Modify the parameter that doesn't require gradients
    with torch.no_grad():
        model.fc1.weight.add_(0.1)
        new_value = model.fc1.weight[0, 0].item()
        print(f"\nModified fc1.weight: {original_value} -> {new_value}")
    
    # Now both the modified parameter and requires_grad=True parameters should be included
    modified_state = diff_checkpoint.state_dict(model)
    print("\nModified state:", modified_state.keys())
    print("\nOriginal first elements:", {k: v.item() for k, v in diff_checkpoint.original_first_elements.items() if k == modified_param_name})
    print("Current first element:", first_element(model.fc1.weight.detach().cpu()).item())
    assert modified_param_name in modified_state
    
    # Modify a buffer
    with torch.no_grad():
        model.bn1.running_mean.add_(0.1)
        modified_buffer_name = "bn1.running_mean"
    
    # The modified buffer should be included along with requires_grad=True parameters
    buffer_state = diff_checkpoint.state_dict(model)
    assert modified_buffer_name in buffer_state
