from pathlib import Path
import numpy as np 
from PIL import Image
import cv2
from tqdm import tqdm
import torch

from src.model import FacePointsModel
from src.utils import get_device, load_from_state_dict



class FacePointsDetector():

    """
    Class for face landmarks detection and coordinate regression.

    Attributes:
        input_size (int): Target input resolution for the model (input_size x input_size).
        device (torch.device): Computation device used for inference.
        model (torch.nn.Module): Compiled FacePointsModel loaded on the specified device.
    """

    def __init__(
            self,
            model_path: str | Path,
            input_size: int = 224,
            device: str | torch.device | None = None,
                 ):
        
        """
        Initializes the detector, loads weights, and prepares the model for inference.

        Args:
            model_path (str | Path): Path to the model's state dictionary (.pth file).
            input_size (int): Target resolution to which input images will be resized
                Do not change unless you have trained your custom model with another resolution.
            device (str | torch.device | None): Device to run inference on. 
                If None, automatically selects the best available device.
        """
        
        self.input_size = int(input_size)

        if device is not None:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device

        else:
            self.device = get_device()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.model = FacePointsModel(input_size=input_size)
        load_from_state_dict(self.model, Path(model_path))
        self.model.to(self.device)
        self.model.eval()

        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass

        self.model = self.model.to(memory_format=torch.channels_last)

        dummy = torch.zeros(1, 3, self.input_size, self.input_size).to(device=self.device)

        with torch.inference_mode():
            self.model(dummy)

    
    def preprocess(self, image: np.ndarray | Image.Image):

        """
        Resizes and normalizes the image to the model's input format.

        Args:
            image (np.ndarray | Image.Image): Input image in RGB format.

        Returns:
            tuple[np.ndarray, tuple[int, int]]: A tuple containing:
                - preprocessed_image: Rescaled and normalized image (float32).
                - original_shape: The (height, width) of the image before resizing.
        """

        image = np.asarray(image)
        original_shape = image.shape[:2]
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(np.float32) / 255.0

        return image, original_shape
    

    def postprocess(
            self, 
            pred_tensor: torch.Tensor, 
            original_shape: np.ndarray | list[int] | tuple[int, int]):

        """
        Resizes predicted coordinates back to original image size.

        Args:
            pred_tensor (torch.Tensor): Tensor with predicted coordinates (x_1, y_1, x_2, y_2, ...).
            original_shape(np.ndarray | list[int] | tuple[int, int]): Original size of the image.

        Returns:
            list[float]: List of resized coordinates in pixels.
        """

        y, x = original_shape

        pred_tensor[0::2] *= x / self.input_size
        pred_tensor[1::2] *= y / self.input_size

        pred = pred_tensor.tolist()

        return pred
    
    def _make_batch(
            self, 
            images: list[np.ndarray | Image.Image], 
            names: list[str], 
            batch_num: int, 
            batch_size: int, 
            pin_memory: bool
            ):
        
        """Slices, preprocesses, and converts a chunk of images into a PyTorch batch."""

        batch = []
        original_shapes = []

        images_slice = images[batch_num * batch_size: (batch_num + 1) * batch_size]
        image_names_slice = names[batch_num * batch_size: (batch_num + 1) * batch_size]

        for image in images_slice:

            image, original_shape = self.preprocess(image)
            original_shapes.append(original_shape)
            batch.append(image)

        batch = np.array(batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(dtype=torch.float32, 
                                                               memory_format=torch.channels_last)

        if pin_memory:
            batch = batch.pin_memory()

        return batch, original_shapes, image_names_slice


    def detect(
            self,
            images: list[np.ndarray | Image.Image],
            image_names: list[str],
            batch_size: int = 16,
            use_amp: bool = False,
            pin_memory: bool = False
            ):

        """
        Performs facial landmark detection and returns scaled coordinates.

        Args:
            images (list[np.ndarray | Image.Image]): Input images for inference.
            image_names (list[str]): Names of the images to use as keys in the output dictionary.
            batch_size (int): Number of images processed per batch. 
                Lower values (4, 8) are recommended for devices with limited memory.
            use_amp (bool): Whether to use FP16 mixed precision. 
                Can significantly speed up inference on modern GPUs.
            pin_memory (bool): Whether to use pinned memory for batches. 
                Speeds up CPU-to-GPU data transfer but increases RAM usage.

        Returns:
            dict[str, list[float]]: A dictionary where each key is an image name 
                and the value is a list of predicted [x1, y1, x2, y2, ...] coordinates.
        """
        

        if len(images) != len(image_names):
            raise ValueError('Number of names must match number of images')
        
        result_dict = {}

        n_batches = (len(images) - 1) // batch_size + 1

        device_type = 'cuda' if self.device.type == 'cuda' else \
                      'mps' if self.device.type == 'mps' else \
                      'cpu'

        for batch_num in tqdm(range(n_batches), desc=f'Inference'):

            batch, original_shapes, image_names_slice = self._make_batch(
                images=images, 
                names=image_names,
                batch_num=batch_num, 
                batch_size=batch_size,
                pin_memory=pin_memory)

            with torch.inference_mode(), torch.amp.autocast(device_type=device_type, enabled=use_amp):

                batch = batch.to(device=self.device, non_blocking=pin_memory)

                preds = self.model(batch)
                preds = preds.cpu()

                for pred_tensor, name, original_shape in zip(preds, image_names_slice, original_shapes):

                    pred = pred_tensor.clone()

                    pred = self.postprocess(pred, original_shape)
                    
                    if name not in result_dict:
                        result_dict[name] = pred
                    else:
                        k = 1

                        while f'{name}_{k}' in result_dict:
                            k += 1

                        result_dict[f'{name}_{k}'] = pred

        return result_dict