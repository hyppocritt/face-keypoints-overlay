import numpy as np 
import random
import yaml
import torch
from pathlib import Path


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


def load_from_state_dict(model, state_dict_path: Path | str, device: torch.device = None):

    if device is None:
        device = torch.device('cpu')

    state_dict_path = Path(state_dict_path).resolve()
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    except Exception as e:
        
        try:
            model.load_state_dict(torch.load(state_dict_path, map_location=device)['model_state_dict'])
        except Exception as e:
            print('Could not load model from checkpoint/state dict.')


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)