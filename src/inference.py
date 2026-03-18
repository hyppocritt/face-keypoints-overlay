from pathlib import Path
import numpy as np 
import pandas as pd
from tqdm import tqdm
import torch

from src.detector import FacePointsDetector
from src.utils.common import chunk_list
from src.utils.image import collect_image_paths, load_images
from src.utils.io import save_json
from src.utils.metrics import calculate_metric
from src.utils.visualisation import visualize
from src.utils.settings import Settings

def run_inference(
        data_path: str | Path,
        detector: FacePointsDetector = None,
        weights_dir: str | Path = None,
        model_args: dict = None,
        model_path: str | Path = None,
        detect_args: dict = None,
        chunk_size: int = 256,
        save: bool = False,
        save_dir: str | Path = './output',
        gt_path: str | Path = None,
        metric: str | None = None,
        vis: str = None,
        save_vis: bool = False,
):
    
    """
    Inference script that runs whole image processing pipeline.

    Args:
        data_path (str | Path): Path to directory that contains images inside or in its subdirectories.
        detector (FacePointsDetector): Detector object to use. If None, one is created.
        weights_dir (str | Path): Path to a directory with model weights. Provide if no Detector were given.
        model_args (dict): Dictionary of arguments used in model initialization.
        model_path (str | Path): Path to models state dictionary.
        save (bool): Whether to save results as JSON file.
        save_dir (str | Path): Where to save results. Default to "./output".
        gt_path (str | Path): Path to a file with ground truth coordinates in format [name, x1, y1, x2, y2, ...].
        metric (str | None): Type of metric to calculate. Supports "mse" and "rmse". Default to None(no metric).
        vis (str): Whether to show visualisation on screen. Supports "all" to show visualization for every processed image, "first" to show for first image only. Default to None(do not show).
        save_vis (bool): Wether to save visualization in saving directory.

    Returns:
        dict[list]: Dictionary with image names as keys and predicted cordinates as values.

    Raises:
        FileNotFoundError: If invalid paths to model weights' directory, config directory or ground truth file are given.
        RuntimeError: If metric is chosen but no ground truth provided.
        ValueError: If unsupported ground truth file type is given.
    """

    data_path = Path(data_path).resolve()
        
    if save or save_vis:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    if vis not in {None, 'first', 'all'}:
        raise ValueError('vis must be one of: None, "first", "all"')

    if metric is not None:
        if gt_path is None:
            raise RuntimeError('Metric was chosen but no ground truth provided.')

        
        gt_path = Path(gt_path).resolve()
        if not gt_path.is_file():
            raise FileNotFoundError('Ground truth file does not exist.')
        
        if gt_path.suffix == '.csv':
            gt_df = pd.read_csv(gt_path)
        elif gt_path.suffix == '.xlsx':
            gt_df = pd.read_excel(gt_path)
        elif gt_path.suffix == '.json':
            gt_df = pd.read_json(gt_path)
        else:
            raise ValueError(f'Unknown ground truth file extension: {gt_path.suffix}')
        

    if detector is None:
        if model_path is None:
            if (model_args is not None) and model_args:
                model_path = Path(weights_dir).resolve() / model_args['filename']
        if (not model_path) or (not model_path.exists()):
            raise FileNotFoundError('Can not find model weights to create detector. Provide detector or model weights path.')
        
        detector = FacePointsDetector(
            model_path=model_path,
            input_size=model_args['input_size']
        )

    if detect_args is None:
        detect_args = {}

    
    image_paths = collect_image_paths(data_path)

    if not image_paths:
        raise RuntimeError('No images found.')

    results = {}
    
    for chunk in tqdm(chunk_list(image_paths, size=chunk_size), 
                      total=(len(image_paths) + 1) // chunk_size):

        images = load_images(chunk)
        image_names = [p.stem for p in chunk]

        current_results = detector.detect(
                            images=images,
                            image_names=image_names, 
                            **detect_args
                            )
        results.update(current_results)
            

        if vis or save_vis:

            for i, name in enumerate(image_names):

                image = images[i]
                coords = current_results[name]

                vis_fig = (i == 0 and vis == 'first') \
                        or (vis == 'all')

                visualize(
                    image=image,
                    coords=coords,
                    name=name,
                    show=vis_fig,
                    save=save_vis,
                    save_dir=(save_dir / 'img') if save_vis else None
                    )

    if save:
        save_json(results, save_dir / 'results.json')

    if metric is not None:

        gt_df.columns = ['name'] + [f'{ax}{k}' for k in range(1, 29) for ax in ('x', 'y')]
        gt_df.set_index('name', inplace=True)

        metrics = {}
        total = []

        for name, coords in results.items():
            
            m = calculate_metric(gt_df.loc[name], coords, metric=metric)
            total.append(m)
            metrics[name] = m

        total_mean = float(np.mean(total))
        metrics[f'mean_{metric}'] = total_mean

        save_json(metrics, save_dir / 'metrics.json')

            
    return results


def main(settings: Settings):
    
    data_path = settings.data.path

    try:
        model_path = settings.resolve_model_path()
    except ValueError as e:
        raise RuntimeError('Can not find model weights. Please provide model weights\' path.') from e
    
    model_path = Path(model_path).resolve()
    
    device = getattr(settings.detector, 'device', None)

    detector = FacePointsDetector(
        model_path=model_path,
        input_size=settings.model.input_size,
        device=device
        )

    run_inference(
        data_path=data_path,
        detector=detector,
        chunk_size=settings.data.chunk_size,
        detect_args=settings.detect,
        **settings.inference
        )