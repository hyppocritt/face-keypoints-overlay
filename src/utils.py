import numpy as np 
import random
import yaml
import json
import torch
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Sequence, Generator



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
    


def collect_image_paths(path: str | Path) -> list[Path]:


    """
    Collects paths to all supported images in a directory and its subdirectories.

    If the provided path points to a single image file, the function returns a list
    containing only that path.

    Args:
        path (str | Path): Path to an image file or a directory.

    Returns:
        list[Path]: Sorted list of image paths with supported extensions.

    Raises:
        FileNotFoundError: if invalid path is provided.
    """

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    supported = {'.png', '.jpg', '.jpeg'}

    if path.is_file():
        return [path]

    return sorted(
        p for p in path.rglob('*')
        if p.suffix.lower() in supported
    )
            


def load_images(paths: list[Path]) -> list[Image.Image]:

    """
    Loads images from given paths.

    Args:
        paths (list[Path]): List of image paths to load.

    Returns:
        list[Image.Image]: List of loaded PIL Image objects.
    """

    res = []

    for path in paths:

            with Image.open(path)as image:
                res.append(ImageOps.exif_transpose(image.copy()))

    return res



def calculate_metric(
        keypoints_gt: list | np.ndarray, 
        keypoints_pred: list | np.ndarray, 
        metric: str = 'mse'
        ) -> float:
    
    """
    Calculates a metric between ground truth and predicted keypoint coordinates.

    Args:
        keypoints_gt (list | np.ndarray): Ground truth keypoint coordinates.
        keypoints_pred (list | np.ndarray): Predicted keypoint coordinates.
        metric (str): Metric to compute. Supported values: "mse", "rmse".

    Returns:
        float: Calculated metric value.

    Raises:
        ValueError: If an unsupported metric is specified.
    """
    
    keypoints_gt = np.asarray(keypoints_gt)
    keypoints_pred = np.asarray(keypoints_pred)

    if metric == 'mse':
        
        return np.mean((keypoints_gt - keypoints_pred) ** 2)
    
    elif metric == 'rmse':

        return np.sqrt(np.mean((keypoints_gt - keypoints_pred) ** 2))
    
    else:
        raise ValueError(f'Unknown metric: {metric}')
    


def save_fig(fig: Figure, save_dir: str | Path, name: str = 'figure'):

    """
    Saves a matplotlib figure to the specified directory.

    If a file with the given name already exists, an incremental suffix
    (_1, _2, ...) is added to avoid overwriting.

    Args:
        fig (Figure): Matplotlib figure to save.
        save_dir (str | Path): Directory where the figure will be saved.
        name (str): Preferred filename without extension. Defaults to "figure".
    """

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f'{name}.png'

    k = 1
    while path.exists():
        path = save_dir / f'{name}_{k}.png'
        k += 1

    fig.savefig(path)



def visualize(
        image: np.ndarray | Image.Image, 
        coords: list | np.ndarray,
        name: str = None,
        show: bool = False,
        save: bool = True,
        save_dir: str | Path = './pic/'
        ):
    
    """
    Visualizes predicted keypoints by overlaying them on the input image.

    Args:
        image (np.ndarray | Image.Image): Original image.
        coords (list | np.ndarray): Predicted keypoint coordinates
            in the format [x1, y1, x2, y2, ...].
        name (str | None): Name used when saving the figure.
        show (bool): Whether to display the figure.
        save (bool): Whether to save the figure to disk.
        save_dir (str | Path): Directory where the figure will be saved.

    Returns:
        Figure: The created matplotlib figure.
    """
    
    coords = np.asarray(coords)

    x_s = coords[0::2]
    y_s = coords[1::2]

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(image, label='original')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(image, label='predicted keypoints')
    axs[1].scatter(x_s, y_s, c='C0', s=10)
    axs[1].set_title('Predicted keypoints')
    axs[1].axis('off')

    if save:

        save_fig(fig, save_dir, name)

    if show:
        plt.show()

    return fig



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

    
def save_json(dct: dict, path: str | Path, indent: int = 2):

    path = Path(path).resolve()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dct, f, indent=indent)