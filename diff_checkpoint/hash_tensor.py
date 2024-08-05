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
    # Convert tensor to float32 if it is bfloat16 to avoid TypeError
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return hashlib.sha256(tensor.cpu().detach().numpy().tobytes()).hexdigest()
