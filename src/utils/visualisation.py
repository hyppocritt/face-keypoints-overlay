import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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