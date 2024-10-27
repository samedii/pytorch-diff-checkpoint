import time
import torch
from torchvision import models
from diff_checkpoint.first_element import first_element  # Updated import


def test_first_element_performance():
    """
    This test checks the performance of the first_element function by measuring the time taken to process a large tensor.
    """
    large_tensor = torch.randn(1000, 1000, 100)  # Create a large tensor

    start_time = time.time()
    result = first_element(large_tensor)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken to process the large tensor: {elapsed_time:.4f} seconds")

    assert isinstance(result, torch.Tensor), "Result should be a tensor"
    assert result.numel() == 1, "Result should be a scalar tensor"


def test_first_element_performance_small_network():
    """
    This test checks the performance of the first_element function by measuring the time taken to process a standard smaller network.
    """
    model = models.resnet18(pretrained=False)
    model.eval()

    start_time = time.time()
    for param in model.parameters():
        result = first_element(param)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken to process the small network: {elapsed_time:.4f} seconds")

    assert isinstance(result, torch.Tensor), "Result should be a tensor"
    assert result.numel() == 1, "Result should be a scalar tensor"
