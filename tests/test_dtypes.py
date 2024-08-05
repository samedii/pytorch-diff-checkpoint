import pytest
import torch
import tempfile

from diff_checkpoint import DiffCheckpoint


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_diff_checkpoint_with_various_dtypes(dtype):
    """
    This test checks that the DiffCheckpoint can handle various tensor dtypes.
    """

    class SimpleModelWithDtype(torch.nn.Module):
        def __init__(self, dtype):
            super(SimpleModelWithDtype, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10, dtype=dtype)
            self.fc2 = torch.nn.Linear(10, 1, dtype=dtype)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModelWithDtype(dtype)
    diff_checkpoint = DiffCheckpoint.from_base_model(model)

    # Modify the weights to ensure there are changes to save
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))

    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(temp_file.name)
        saved_diff_checkpoint = torch.load(temp_file.name, weights_only=True)
        assert (
            len(saved_diff_checkpoint) > 0
        ), "Checkpoint should contain modified parameters"
