import torch
from torch.autograd import grad
import math


def normed_dot_product(x1: torch.Tensor, x2: torch.Tensor):    # normalize vector for correct angle
    unit_x1 = x1 / torch.norm(x1)
    unit_x2 = x2 / torch.norm(x2)
    return torch.dot(unit_x1, unit_x2)


def angle(x1: torch.Tensor, x2: torch.Tensor):
    dot_product = normed_dot_product(x1, x2)
    return (torch.acos(dot_product) / math.pi) * 180



def loss_gradients(loss, model, use_gpu=True):
    """
    Computes the gradient of a loss wrt. to the model parameters

    parms:
    loss: loss term
    model: model
    """
    device = torch.device("cuda" if use_gpu else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    model_grads = grad(loss, model.parameters(), allow_unused=True, retain_graph=True)
    for elem in model_grads:
        if elem is not None:
            grad_ = torch.cat((grad_, elem.view(-1)))
    return grad_


def capture_gradients(model):
    """
    Concatenates gradients of a model after calling backward


    return tensor of gradients
    """
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    return torch.cat(grads)
