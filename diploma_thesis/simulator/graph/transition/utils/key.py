import torch


def key(value: torch.Tensor | int):
    if isinstance(value, int):
        return str(value)

    if torch.is_tensor(value):
        return str(value.item())

    return str(value)

def unkey(value, is_tensor: bool = False):
    value = int(value)

    if is_tensor:
        return torch.tensor(value)

    return value
