import random
import numpy as np 
import torch
from typing import Generator, Sequence

def get_device():
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else: 
        return torch.device('cpu')
    

def set_seed(seed: int = 78):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def chunk_list(lst: Sequence, size: int = 256) -> Generator[Sequence, None, None]:

    """
    Splits an iterable into consecutive chunks of the given size.

    Args:
        lst (Sequence): Input sequence to split into chunks.
        size (int): Maximum size of each chunk.

    Yields:
        Sequence: Consecutive chunks of the input sequence.
    """

    for i in range(0, len(lst), size):
        yield lst[i: i + size]

