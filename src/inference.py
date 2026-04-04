from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.detector import FacePointsDetector
from src.utils.common import chunk_list
from src.utils.image import collect_image_paths, load_images
from src.utils.io import save_json
from src.utils.metrics import calculate_metric
from src.utils.settings import Settings
from src.utils.visualization import get_visualization_callback


def run_inference(
    data_path: str | Path | None = None,
    images: list[np.ndarray | Image.Image] | None = None,
    image_names: list[str] | None = None,
    detector: FacePointsDetector = None,
    weights_dir: str | Path = None,
    model_args: dict = None,
    model_path: str | Path = None,
    detect_args: dict = None,
    chunk_size: int = 256,
    save: bool = False,
    save_dir: str | Path = "./output",
    gt_path: str | Path = None,
    metric: str | None = None,
    on_chunk_end: Callable | list[Callable] = None,
) -> dict[str, list[float]]:
    """
    Inference script that runs whole image processing pipeline.

    Args:
        data_path (str | Path | None): Path to directory that contains images inside or in its subdirectories.
        images (list[np.ndarray | Image.Image] | None): List of already loaded images.
                                                        Alternative source.
        image_names (list[str | None): List of names for provided images.
                                       Will be ignored if data_path is provided.
        detector (FacePointsDetector): Detector object to use. If None, one is created.
        weights_dir (str | Path): Path to a directory with model weights. Provide if no Detector were given.
        model_args (dict): Dictionary of arguments used in model initialization.
        model_path (str | Path): Path to models state dictionary.
        detect_args (dict): Dictionary of arguments used in detection.
        chunk_size (int): Size of a chunk of images passed to detector at a time.
                          If data_path is provided, will limit RAM usage.
        save (bool): Whether to save results as JSON file.
        save_dir (str | Path): Where to save results. Default to "./output".
        gt_path (str | Path): Path to a file with ground truth coordinates in format [name, x1, y1, x2, y2, ...].
        metric (str | None): Type of metric to calculate. Supports "mse" and "rmse". Default to None(no metric).
        on_chunk_end (Callable): Callback used when images chunk is processed.

    Returns:
        dict[str, list[float]]: Dictionary with image names as keys and predicted cordinates as values.

    Raises:
        FileNotFoundError: If invalid paths to model weights' directory, config directory or ground truth file are given.
        RuntimeError: If metric is chosen but no ground truth provided.
        ValueError: If unsupported ground truth file type is given.
    """

    def process_chunk(
        images_chunk: list[Image.Image | Path],
        names_chunk: list[str],
        results: dict[str, list[float]],
        callbacks: list[Callable],
    ):

        if isinstance(images_chunk[0], Path):
            images_chunk = load_images(images_chunk)

        results_chunk = detector.detect(
            images=images_chunk, image_names=names_chunk, **detect_args
        )
        results.update(results_chunk)

        for callback in callbacks:
            callback(
                images_chunk,
                names_chunk,
                results_chunk,
            )

    if data_path is None and images is None:
        raise ValueError("Provide either path to data or images list.")

    if data_path is not None and images is not None:
        raise ValueError(
            "Only one source should be provided, but got both data_path and images list"
        )

    if save:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    if metric is not None:
        if gt_path is None:
            raise RuntimeError("Metric was chosen but no ground truth provided.")

        gt_path = Path(gt_path).resolve()
        if not gt_path.is_file():
            raise FileNotFoundError("Ground truth file does not exist.")

        if gt_path.suffix == ".csv":
            gt_df = pd.read_csv(gt_path)
        elif gt_path.suffix == ".xlsx":
            gt_df = pd.read_excel(gt_path)
        elif gt_path.suffix == ".json":
            gt_df = pd.read_json(gt_path)
        else:
            raise ValueError(f"Unknown ground truth file extension: {gt_path.suffix}")

    if detector is None:
        if model_path is None:
            if (model_args is not None) and model_args:
                model_path = Path(weights_dir).resolve() / model_args["filename"]

        model_path = Path(model_path).resolve()
        if (not model_path) or (not model_path.exists()):
            raise FileNotFoundError(
                "Can not find model weights to create detector. Provide detector or model weights path."
            )

        detector = FacePointsDetector(
            model_path=model_path, input_size=model_args["input_size"]
        )

    if detect_args is None:
        detect_args = {}

    if on_chunk_end is None:
        on_chunk_end = []
    elif not isinstance(on_chunk_end, (list, tuple)):
        on_chunk_end = [on_chunk_end]
    else:
        on_chunk_end = list(on_chunk_end)

    on_chunk_end = [cb for cb in on_chunk_end if cb is not None]

    results = {}

    if data_path is not None:
        data_path = Path(data_path).resolve()

        image_paths = collect_image_paths(data_path)

        if not image_paths:
            raise RuntimeError(f"No images found at {data_path}.")

        chunks = tqdm(
            zip(
                chunk_list(image_paths, size=chunk_size),
                chunk_list([p.stem for p in image_paths], size=chunk_size),
            ),
            total=(len(image_paths) + 1) // chunk_size,
        )

        for images_chunk, names_chunk in chunks:
            process_chunk(
                images_chunk=images_chunk,
                names_chunk=names_chunk,
                results=results,
                callbacks=on_chunk_end,
            )

    elif images is not None:
        if not images:
            raise RuntimeError("No images found in image list.")

        if isinstance(images, (np.ndarray, Image.Image)):
            images = [images]
        else:
            images = list(images)

        if image_names is None:
            image_names = [f"img_{i}" for i in range(len(images))]

        if isinstance(image_names, str):
            image_names = [image_names]
        else:
            image_names = list(image_names)

        if len(image_names) != len(images):
            raise ValueError(
                f"Got {len(images)} images, but only {len(image_names)} names. \
                               Provide valid names list or use None for automatic naming."
            )

        chunks = tqdm(
            zip(
                chunk_list(images, size=chunk_size),
                chunk_list(image_names, size=chunk_size),
            ),
            total=(len(images) + 1) // chunk_size,
        )

        for images_chunk, names_chunk in chunks:
            process_chunk(
                images_chunk=images_chunk,
                names_chunk=names_chunk,
                results=results,
                callbacks=on_chunk_end,
            )

    if save:
        save_json(results, save_dir / "results.json")

    if metric is not None:
        gt_df.columns = ["name"] + [
            f"{ax}{k}" for k in range(1, 29) for ax in ("x", "y")
        ]
        gt_df.set_index("name", inplace=True)

        metrics_dict = {}
        total = []
        total_mean = {}

        for name, coords in results.items():
            m = calculate_metric(gt_df.loc[name], coords, metric=metric)
            total.append(m)
            metrics_dict[name] = m

        metrics = [metric] if isinstance(metric, str) else metric

        for mtrc in metrics:
            total_metric = [m[mtrc] for m in total]
            total_mean[mtrc] = float(np.mean(total_metric))
        metrics_dict["mean"] = total_mean

        save_json(metrics_dict, save_dir / "metrics.json")

    return results


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

    visualization_callback = get_visualization_callback(
        vis=settings.visualization.vis,
        save_vis=settings.visualization.save_vis,
        save_dir=settings.visualization.save_dir,
    )

    run_inference(
        data_path=data_path,
        detector=detector,
        chunk_size=settings.data.chunk_size,
        detect_args=settings.detect,
        on_chunk_end=visualization_callback,
        **settings.inference,
    )
