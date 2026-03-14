from pathlib import Path
from PIL import Image, ImageOps

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