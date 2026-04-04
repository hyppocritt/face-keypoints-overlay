from src.detector import FacePointsDetector
from src.inference import run_inference
from src.utils.visualization import get_visualization_callback


def test_inference(data_path, detector, callback):

    results = run_inference(
        data_path=data_path, detector=detector, save=False, on_chunk_end=callback
    )

    assert isinstance(results, dict)


if __name__ == "__main__":
    data_path = "./tests/data/sample_image.jpg"
    model_type = "resnet"
    model_path = "./weights/facepoints_resnet_nme_0_0469.pth"

    detector = FacePointsDetector(model_path=model_path, model_type=model_type)

    cb = get_visualization_callback(
        vis="first",
        save_vis=False,
    )

    test_inference(data_path, detector, cb)
