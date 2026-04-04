import json
from pathlib import Path

import torch
import yaml


def load_from_state_dict(
    model, state_dict_path: Path | str, device: torch.device = None
):

    if device is None:
        device = torch.device("cpu")

    state_dict_path = Path(state_dict_path).resolve()
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    except Exception:
        try:
            model.load_state_dict(
                torch.load(state_dict_path, map_location=device)["model_state_dict"]
            )
        except Exception:
            print("Could not load model from checkpoint/state dict.")


def load_yaml(path):

    with open(path) as f:
        return yaml.safe_load(f)


def read_json(path: str | Path) -> dict:

    path = Path(path).resolve()
    if not path.exists():
        raise RuntimeError(f"Path does not exist: {path}.")

    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result


def save_json(dct: dict, path: str | Path, indent: int = 2):

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dct, f, indent=indent)
