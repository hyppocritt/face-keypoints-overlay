from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.detector import FacePointsDetector
from src.inference import run_inference
from src.mask import FaceMask
from src.utils.settings import Settings
from src.utils.visualization import visualize

MASK_CACHE = {}


def get_mask(name: str) -> FaceMask:

    if name not in MASK_CACHE:
        MASK_CACHE[name] = FaceMask(name)

    return MASK_CACHE[name]


def clear_mask_cache():

    global MASK_CACHE
    MASK_CACHE.clear()


def apply_overlay(
    images: Image.Image | np.ndarray | list[Image.Image] | list[np.ndarray],
    keypoints: list[float] | dict[list[float]],
    image_names: str | list[str] | None = None,
    mask: str = "glasses_and_mustache",
    save: bool = True,
    save_dir: str | Path = "./output/",
    vis: str = None,
    save_vis: bool = False,
):

    if images is None:
        raise ValueError("Images must be provided.")

    if isinstance(images, np.ndarray) or isinstance(images, Image.Image):
        images = [np.asarray(images)]
    else:
        images = list(images)

    if image_names is None:
        image_names = [f"img_{i}" for i in range(len(images))]

    if isinstance(image_names, str):
        image_names = [image_names]

    if len(image_names) != len(images):
        raise ValueError(
            f"Got {len(images)} images, but only {len(image_names)} names. \
                               Provide valid names list or use None for automatic naming."
        )

    if keypoints is None:
        raise ValueError("Keypoints must be provided.")

    if isinstance(keypoints, (list, tuple)):
        keypoints = {image_names[0]: list(keypoints)}

    if len(keypoints) != len(images):
        raise ValueError(
            f"Got {len(images)} images, but {len(keypoints)} set of keypoits coordinates."
        )

    if save_dir is not None:
        save_dir = Path(save_dir).resolve()
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

    mask = get_mask(mask)

    results = []

    for i, (image, name) in enumerate(
        tqdm(zip(images, image_names), desc="Applying overlay", total=len(images))
    ):
        result = mask.apply(image=image, keypoints=keypoints[name])

        result = Image.fromarray(result)

        results.append(result)

        if save:
            save_path = save_dir / f"{name}_edited.jpg"

            if save_path.exists():
                k = 1
                while (save_dir / f"{name}_{k}_edited.jpg").exists():
                    k += 1
                save_path = save_dir / f"{name}_{k}_edited.jpg"

            result.save(save_path)

        if vis or save_vis:
            vis_fig = (vis == "first" and i == 0) or (vis == "all")

            visualize(
                image=image,
                coords=keypoints[name],
                result=result,
                name=name,
                show=vis_fig,
                save=save_vis,
                save_dir=(save_dir / "img") if save_vis else None,
            )

    return results


def get_overlay_callback(
    mask: str = "glasses_and_mustache",
    save: bool = True,
    save_dir: str | Path = "./output/",
    vis: str = None,
    save_vis: bool = False,
):

    def overlay_wrapper(images_chunk, names_chunk, result_chunk):
        return apply_overlay(
            images=images_chunk,
            image_names=names_chunk,
            keypoints=result_chunk,
            mask=mask,
            save=save,
            save_dir=save_dir,
            vis=vis,
            save_vis=save_vis,
        )

    return overlay_wrapper


def main(settings: Settings):

    data_path = settings.data.path

    try:
        model_path = settings.resolve_model_path()
    except ValueError as e:
        raise RuntimeError(
            "Can not find model weights. Please provide model weights' path."
        ) from e

    model_path = Path(model_path).resolve()

    device = getattr(settings.detector, "device", None)

    detector = FacePointsDetector(
        model_path=model_path,
        model_type=settings.model.model_type,
        input_size=settings.model.input_size,
        device=device,
    )

    overlay_callback = get_overlay_callback(
        mask=settings.overlay.mask,
        save=settings.overlay.save,
        vis=settings.overlay.vis,
        save_vis=settings.overlay.save_vis,
        save_dir=settings.overlay.save_dir,
    )

    run_inference(
        data_path=data_path,
        detector=detector,
        chunk_size=settings.data.chunk_size,
        detect_args=settings.detect,
        on_chunk_end=overlay_callback,
        **settings.inference,
    )
