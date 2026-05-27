import torch
import torch.nn as nn
import numpy.typing as npt
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    max_start = len(dataset)-context_length-1

    starts = np.random.randint(0, max_start+1, size=batch_size)

    '''
    x_batch = []
    y_batch = []
    
    for start in starts:
        x = dataset[start:start+context_length]
        y = dataset[start+1: start+1+context_length]

        x_batch.append(x)
        y_batch.append(y)


    x_tensor = torch.tensor(np.array(x_batch), dtype=torch.long, device=device)
    y_tensor = torch.tensor(np.array(y_batch), dtype=torch.long, device=device)


    return x_tensor, y_tensor
    '''

    x = np.stack([
        dataset[start : start + context_length]
        for start in starts
    ])

    y = np.stack([
        dataset[start + 1 : start + 1 + context_length]
        for start in starts
    ])

    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)

    return x, y
