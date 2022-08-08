import torch


def angle(x1: torch.Tensor, x2: torch.Tensor):
    # normalize vector for correct angle
    unit_x1 = x1 / torch.norm(x1)
    unit_x2 = x2 / torch.norm(x2)

    dot_product = torch.dot(unit_x1, unit_x2)
    return (torch.arccos(dot_product) / torch.pi) * 180
