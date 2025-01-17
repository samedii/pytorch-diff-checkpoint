import tempfile

import pytest
import torch
from torchvision import models

from diff_checkpoint import DiffCheckpoint


def test_vram_usage():
    """
    This test checks that DiffCheckpoint has minimal VRAM overhead.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping VRAM usage test")
    
    # Record initial VRAM usage
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    # Create a large model
    model = models.resnet152(pretrained=False).cuda()
    model_memory = torch.cuda.memory_allocated() - initial_memory
    print(f"Model VRAM usage: {model_memory / 1024**2:.2f} MB")

    # Create checkpoint and record memory
    before_checkpoint = torch.cuda.memory_allocated()
    diff_checkpoint = DiffCheckpoint.from_base_model(model)
    after_checkpoint = torch.cuda.memory_allocated()
    checkpoint_overhead = after_checkpoint - before_checkpoint
    print(f"Checkpoint creation overhead: {checkpoint_overhead / 1024**2:.2f} MB")
    
    # The overhead should be minimal (allow for small allocations)
    assert checkpoint_overhead <= 1024 * 1024, "Checkpoint creation should have minimal VRAM overhead"

    # Modify some weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(0.1)

    # Save checkpoint and record memory
    before_save = torch.cuda.memory_allocated()
    with tempfile.NamedTemporaryFile() as temp_file:
        diff_checkpoint.save(model, temp_file.name)
    after_save = torch.cuda.memory_allocated()
    save_overhead = after_save - before_save
    print(f"Checkpoint saving overhead: {save_overhead / 1024**2:.2f} MB")
    
    # Allow for some temporary allocations during save
    assert save_overhead <= 1024 * 1024, "Checkpoint saving should have minimal VRAM overhead" 