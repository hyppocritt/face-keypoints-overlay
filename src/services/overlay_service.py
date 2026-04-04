from PIL import Image
from pathlib import Path

from src.detector import FacePointsDetector
from src.inference import run_inference
from src.overlay import apply_overlay
from src.utils.settings import Settings


class OverlayService():
    def __init__(
            self,
            model_path: str | Path,
            model_type: str,
            batch_size: int,
            chunk_size: int,
            save_keypoints: bool,
            save_overlays: bool,
            ):
        
        self.detector = FacePointsDetector(
            model_path=model_path,
            model_type=model_type,
            batch_size=batch_size
        )
        self.save_keypoints = save_keypoints
        self.save_overlays = save_overlays
        self.chunk_size = chunk_size


    def process_image(self, image: Image.Image, mask_name: str):
        keypoints_dict = run_inference(
            images=[image],
            image_names=["input"],
            detector=self.detector,
            save=self.save_keypoints,
            chunk_size=self.chunk_size
        )

        keypoints = keypoints_dict["input"]

        result = apply_overlay(
            images=image,
            keypoints=keypoints,
            mask=mask_name,
            save=self.save_overlays,
        )

        return result[0]
    

def create_overlay_service(settings: Settings) -> OverlayService:

    return OverlayService(
        model_path=settings.resolve_model_path(),
        model_type=settings.model.type,
        batch_size=settings.detect.batch_size,
        chunk_size=settings.inference.chunk_size,
        save_keypoints=settings.inference.save,
        save_overlays=settings.overlay.save
    )
