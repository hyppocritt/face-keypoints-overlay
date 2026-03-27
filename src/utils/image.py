from pathlib import Path
from PIL import Image, ImageOps
import numpy as np


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


def alpha_blend(
          image: Image.Image | np.ndarray,
          mask: Image.Image | np.ndarray,
          alpha_map: np.ndarray = None,
          alpha: float = 0.15
) -> np.ndarray:
    
    """
    Blends RGB or Grayscale images using alpha_map(usually A channel from RGBA mask) of just alpha parameter.

    Args:
        image (Image.Image | np.ndarray): Background image.
        mask (Image.Image | np.ndarray): Foreground image.
        alpha_map: (np.ndarray): Map of the same spatial shape as image and mask. 
                                 Stores specific alpha parameter for every point.
        alpha (float): Simple opacity parameter for blending.

    Returns:
        np.ndarray: Result image.

    Raises:
        ValueError: If image of mask have unsupported number of channels.
    """
    
     
    image = np.asarray(image).astype(np.float32)
    mask = np.asarray(mask).astype(np.float32)

    if image.max() > 1:
        image /= 255.0
    if mask.max() > 1:
        mask /= 255.0
    
    if image.ndim == 2:
        image = image[..., None]
    if mask.ndim == 2:
        mask = mask[..., None]

    if alpha_map is None:
        alpha_map = np.full((*image.shape[:2], 1), alpha, dtype=np.float32)

    else:
        alpha_map = np.asarray(alpha_map).astype(np.float32)

        if alpha_map.ndim == 2:
            alpha_map = alpha_map[..., None]
    
    alpha_map = np.clip(alpha_map, 0.0, 1.0)

    if mask.shape[-1] != image.shape[-1]:
        mask = np.repeat(mask, image.shape[-1], axis=-1)

    if alpha_map.shape[-1] == 1:
        alpha_map = np.repeat(alpha_map, image.shape[-1], axis=-1)
        
    result = alpha_map * mask + (1 - alpha_map) * image

    result = (result * 255).clip(0, 255).astype(np.uint8)

    return result
    