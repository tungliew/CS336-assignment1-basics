import torch.nn
import torch.nn as nn

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    # compute the l2_norm
    total = torch.zeros(1)

    for param in parameters:
        if param.grad is None:
            continue
        
        total += torch.sum(param.grad.data ** 2)

    l2_norm = torch.sqrt(total)


    # clipping
    max_l2_norm = torch.tensor(max_l2_norm)
    eps = 1e-6

    if l2_norm>max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            
            param.grad.data = param.grad.data * scale