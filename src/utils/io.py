import json
import yaml
from pathlib import Path
import torch


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
            

def load_yaml(path):

    with open(path) as f:
        return yaml.safe_load(f)


def save_json(dct: dict, path: str | Path, indent: int = 2):

    path = Path(path).resolve()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dct, f, indent=indent)