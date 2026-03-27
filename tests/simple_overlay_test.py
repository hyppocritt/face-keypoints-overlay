from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from src.inference import run_inference
from src.overlay import get_overlay_callback


def test_overlay(
        image_path: str | Path,
        model_path: str | Path
        ):

    image_path = Path(image_path).resolve()
    model_path = Path(model_path).resolve()

    overlay_cb = get_overlay_callback(
        save=False,
        vis='first'
    )

    coords_dict = run_inference(
        data_path=image_path,
        model_path=model_path,
        model_args={
            'input_size': 224
        },
        on_chunk_end=overlay_cb
    )

    assert isinstance(coords_dict, dict)


if __name__ == '__main__':

    image_path = Path('./tests/data/sample_image.jpg').resolve()
    model_path = Path('./weights/facepoints_resnet_nme_0_0469.pth').resolve()

    test_overlay(image_path, model_path)
