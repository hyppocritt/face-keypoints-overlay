import os 
from dotenv import load_dotenv
from pathlib import Path
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm

from src.detector import FacePointsDetector
from src.utils.common import chunk_list
from src.utils.image import collect_image_paths, load_images
from src.utils.io import load_config, save_json
from src.utils.metrics import calculate_metric
from src.utils.visualisation import visualize

def run_inference(
        data_path: str | Path,
        detector: FacePointsDetector = None,
        weights_dir: str | Path = None,
        config: dict = None,
        config_dir: str | Path = None,
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
        data_path (str | Path): Path to directory that contains images inside or in its' subdirectories.
        detector (FacePointsDetector): Detector object to use. If None, one is created.
        weights_dir (str | Path): Path to a directory with model weights. Provide if no Detector were given.
        config_dir (str | Path): Path to a directory with config files.
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
    
    if config is None:
        if config_dir is None:
            config_dir = Path(os.getenv('CONFIG_DIR', './configs/')).resolve()
        if not config_dir.exists():
            raise FileNotFoundError('No valid model config directory provided.')
        
        config = load_config(config_dir / 'inference.yaml')
        
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
        

    input_size = config['model']['input_size']
    chunk_size = config['data']['chunk_size']

    if detector is None:
        if weights_dir is None:
            weights_dir = Path(os.getenv('WEIGHTS_DIR', './weights/')).resolve()
        if not weights_dir.exists():
            raise FileNotFoundError('Can not find model weights to create detector. Provide detector or model weights path.')
        
        model_path = weights_dir / config['model']['filename']
        detector = FacePointsDetector(
            model_path=model_path,
            input_size=input_size
        )

    
    image_paths = collect_image_paths(data_path)

    if not image_paths:
        raise RuntimeError('No images found')

    results = {}
    
    for chunk in tqdm(chunk_list(image_paths, size=chunk_size), 
                      total=(len(image_paths) + 1) // chunk_size):

        images = load_images(chunk)
        image_names = [p.stem for p in chunk]

        current_results = detector.detect(
                            images=images,
                            image_names=image_names, 
                            **config['detect']
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




    
def get_args():

    """
    Parses arguments for inference from command line.

    Returns:
        dict: Dict of inference keyword arguments.
    """

    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--data_path', type=str, default=None, help='Path to images')
    parser.add_argument('--weights_dir', type=str, default=None, help='Path to a directory with model weights')
    parser.add_argument('--config_dir', type=str, default=None, help='Path to a directory with configs')
    parser.add_argument('--save', action='store_true', default=True, help='Wether to save results')
    parser.add_argument('--save_dir', type=str, default='./output/', help='Path to saving directory')
    parser.add_argument('--gt_path', type=str, default=None, help='Path to file with ground truth coordinates')
    parser.add_argument('--metric', type=str, default=None, help='Type of metric to calculate. Supports "mse", "rmse", None.')
    parser.add_argument('--vis', type=str, choices=['first', 'all'], default=None, help='Wether to show show visualization')
    parser.add_argument('--save_vis', action='store_true', default=False, help='Wether to save visualization')

    return vars(parser.parse_args())


def main():
    
    load_dotenv()
    args = get_args()

    run_inference(**args)