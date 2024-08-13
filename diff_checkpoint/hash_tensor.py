import hashlib

import torch


def hash_tensor(tensor: torch.Tensor) -> str:
    """
    Computes the SHA-256 hash of a tensor.

    Args:
        tensor (torch.Tensor): The tensor to hash.

    Returns:
        str: The SHA-256 hash of the tensor.
    """
    if tensor.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    # Convert tensor to float32 if its dtype is not supported
    supported_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    if tensor.dtype not in supported_dtypes:
        tensor = tensor.to(torch.float32)
    return hashlib.sha256(tensor.cpu().detach().numpy().tobytes()).hexdigest()
