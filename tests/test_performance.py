import time

import torch
from torchvision import models

from diff_checkpoint.hash_tensor import hash_tensor


def test_hash_tensor_performance():
    """
    This test checks the performance of the hash_tensor function by measuring the time taken to hash a large tensor.
    """
    large_tensor = torch.randn(1000, 1000, 100)  # Create a large tensor

    start_time = time.time()
    hash_result = hash_tensor(large_tensor)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken to hash the large tensor: {elapsed_time:.4f} seconds")

    # Ensure the hash result is a string of length 64 (SHA-256 hash length)
    assert isinstance(hash_result, str), "Hash result should be a string"
    assert len(hash_result) == 64, "Hash result should be 64 characters long"


def test_hash_tensor_performance_small_network():
    """
    This test checks the performance of the hash_tensor function by measuring the time taken to hash a standard smaller network.
    """
    model = models.resnet18(pretrained=False)
    model.eval()

    start_time = time.time()
    for param in model.parameters():
        hash_result = hash_tensor(param)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken to hash the small network: {elapsed_time:.4f} seconds")

    # Ensure the hash result is a string of length 64 (SHA-256 hash length)
    assert isinstance(hash_result, str), "Hash result should be a string"
    assert len(hash_result) == 64, "Hash result should be 64 characters long"
