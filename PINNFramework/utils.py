import torch
from torch.autograd import grad
import math
import time

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


def model_gradients(data_points, model, use_gpu=True):
    """
    Computes the gradient of a model output(of initial condition data) wrt. to the model parameters

    parms:
    loss: loss term
    model: model
    """
    start = time.time()
    device = torch.device("cuda" if use_gpu else "cpu")
    model_grads = []
    model_out = model(data_points)

    number_of_output_datapoints = model_out.shape[0]
    number_of_output_labels = model_out.shape[1]

    for i in range(number_of_output_datapoints):
        for j in range(number_of_output_labels):
            grad_ = torch.zeros((0), dtype=torch.float32, device=device)
            connected_grad = grad(model_out[i, j], model.parameters(), allow_unused=True, retain_graph=True)
            # flatten grads in vector
            for elem in connected_grad:
                if elem is not None:
                    grad_ = torch.cat((grad_, elem.view(-1)))
            model_grads.append(grad_)
    end = time.time()
    print("model gradients time: ", end - start)
    return model_grads