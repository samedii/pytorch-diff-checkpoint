# pytorch diff checkpoint

`pytorch-diff-checkpoint` is a PyTorch helper library designed to save only the changes in a fine-tuned base model. This tool is particularly useful for scenarios where you want to minimize storage usage by saving only the parameters that have been modified.

## Installation

```bash
poetry add pytorch-diff-checkpoint
```

## Usage

Create your model.

```python
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

model = SimpleModel()

# Create a DiffCheckpoint from the base model
diff_checkpoint = DiffCheckpoint.from_base_model(model)

# Train
# ...

# Save the differential checkpoint
diff_checkpoint.save('diff_checkpoint.pth')
```
