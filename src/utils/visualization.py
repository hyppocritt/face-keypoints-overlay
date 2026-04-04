from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def save_fig(fig: Figure, save_dir: str | Path, name: str = "figure"):
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

    path = save_dir / f"{name}.png"

    k = 1
    while path.exists():
        path = save_dir / f"{name}_{k}.png"
        k += 1

    fig.savefig(path)


def visualize(
    image: np.ndarray | Image.Image,
    coords: list | np.ndarray,
    result: np.ndarray | Image.Image = None,
    name: str = None,
    show: bool = False,
    save: bool = True,
    save_dir: str | Path = "./pic/",
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

    has_result = result is not None

    coords = np.asarray(coords)

    x_s = coords[0::2]
    y_s = coords[1::2]

    n_cols = 3 if has_result else 2

    fig, axs = plt.subplots(1, n_cols, figsize=(10, 6))
    axs[0].imshow(image, label="original")
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(image, label="predicted keypoints")
    axs[1].scatter(x_s, y_s, c="C0", s=10)
    axs[1].set_title("Predicted keypoints")
    axs[1].axis("off")
    if has_result:
        axs[2].imshow(result, label="overlay")
        axs[2].set_title("Image with overlay")
        axs[2].axis("off")

    plt.suptitle("Pipeline demonstration")

    if save:
        save_fig(fig, save_dir, name)

    if show:
        plt.show()

    return fig


def get_visualization_callback(
    vis: str, save_vis: bool = None, save_dir: str | Path = "./output/"
):
    """
    Returns visualisation callback for inference pipeline based on given params:

    Args:
        vis (str): Whether to show visualisation on screen. Supports "all" to show visualization
                   for every processed image, "first" to show for first image only.
                   Default to None(do not show).
        save_vis (bool): Wether to save visualization in saving directory.
        save_dir (str | Path): Where to save results. Default to "./output".

    Returns:
        Callable: Visualization callback or None

    Raises:
        ValueError: If vis not in [None, 'first', 'all'].
    """

    if vis not in {None, "first", "all"}:
        raise ValueError('vis must be one of: None, "first", "all"')

    if save_dir is None:
        if save_vis:
            raise ValueError(
                "Specify saving directory or use default location, not None."
            )
    else:
        save_dir = Path(save_dir).resolve()

    is_first = True

    def visualize_chunk(
        images_chunk,
        names_chunk,
        results_chunk,
    ):

        nonlocal is_first

        for i, name in enumerate(names_chunk):
            image = images_chunk[i]
            coords = results_chunk[name]

            vis_fig = (is_first and vis == "first") or (vis == "all")

            visualize(
                image=image,
                coords=coords,
                name=name,
                show=vis_fig,
                save=save_vis,
                save_dir=(save_dir / "img") if save_vis else None,
            )

            if vis_fig and vis == "first":
                is_first = False

    return visualize_chunk if (vis is not None) or save_vis else None
