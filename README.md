# pytorch-diff-checkpoint

`pytorch-diff-checkpoint` is a simple library designed to efficiently save only the modified parameters of a fine-tuned base model. This tool is particularly advantageous in scenarios where minimizing storage usage is crucial, as it ensures that only the altered parameters are stored.

## Installation

```bash
poetry add pytorch-diff-checkpoint
```

## Usage

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
diff_checkpoint.save(model, 'diff_checkpoint.pth')
```
