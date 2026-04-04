import pytest

from src.detector import FacePointsDetector
from src.utils.visualization import get_visualization_callback


@pytest.fixture
def data_path():

    data_path = "./tests/data/sample_image.jpg"

    return data_path


@pytest.fixture
def image_path():

    image_path = "./tests/data/sample_image.jpg"

    return image_path


@pytest.fixture
def model_path():

    model_path = "./weights/facepoints_resnet_nme_0_0469.pth"

    return model_path


@pytest.fixture
def detector():

    model_type = "resnet"
    model_path = "./weights/facepoints_resnet_nme_0_0469.pth"

    detector = FacePointsDetector(model_path=model_path, model_type=model_type)

    return detector


@pytest.fixture
def callback():

    callback = get_visualization_callback(
        vis="first",
        save_vis=False,
    )

    return callback
