import torch
import torch.nn as nn
import math

def lr_cosine_schedule(
    t: int,  # t
    alpha_max: float, 
    alpha_min: float, 
    T_w: int, 
    T_c: int, 
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    lr = 0.0

    if t<T_w:
        lr = t * alpha_max / T_w
    elif T_w<=t<=T_c:
        lr = alpha_min + (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min) / 2 
    else:
        lr = alpha_min

    return lr