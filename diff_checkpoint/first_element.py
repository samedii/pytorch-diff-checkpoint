import torch


def first_element(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the first element of a tensor as a scalar tensor.

    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        torch.Tensor: A scalar tensor containing the first element.
    """
    if tensor.numel() == 0:
        return torch.tensor(0.0)  # Return a scalar zero tensor for empty input

    return tensor.flatten()[0]
