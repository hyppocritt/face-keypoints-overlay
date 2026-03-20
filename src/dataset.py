import numpy as np 
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
from pathlib import Path


facepoints_transforms = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=10,
        p=0.5
    ),
],
keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

def flip_coordinates(coords: np.array) -> None:

    flipped_indexes = [6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 16, 17, 14, 15, \
                        12, 13, 10, 11, 8, 9, 20, 21, 26, 27, 24, 25, 22, 23]
    coords[:] = coords[flipped_indexes]

class FacePointsTransformDataset(Dataset):

    """
    Dataset class for face keypoint regression.

    Supports:
    - Albumentation transforms with keypoints
    - Normalization of target coordinates
    """

    def __init__(
            self, image_dir: str, 
            metadata_path: str = None,
            metadata_df: pd.DataFrame = None, 
            image_size: int = 224, 
            normalize_targets: bool = False,
            transforms: A.Compose = facepoints_transforms,
            flip_prob: float = 0.5,
            return_meta: bool = False
            ):

        super().__init__()

        self.image_dir = Path(image_dir).resolve()
        self.has_coords = (metadata_path is not None) or (metadata_df is not None)

        if metadata_df is not None:
            self.metadata = metadata_df
        elif metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:
            extensions = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')

            names = sorted(
                sum(
                    [[p.name for p in self.image_dir.glob(exp) if p.is_file()] for exp in extensions],
                    []
                )
            )
            self.metadata = pd.DataFrame({'filename': names})

        self.image_size = image_size
        self.normalize_targets = normalize_targets
        self.transforms = transforms
        self.flip_prob = flip_prob
        self.return_meta = return_meta
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]
        image_name = row.loc['filename']
        
        image = Image.open(self.image_dir / image_name).convert('RGB')
        orig_w, orig_h = image.size

        image_array = np.asarray(image)
        image_array = image_array.astype(np.float32) / 255.0

        coords_tensor = None

        if self.has_coords:

            coords = row.iloc[1:].to_numpy(dtype=np.float32)

            if self.transforms is not None:

                if np.random.rand() < self.flip_prob:
                    
                    image_array = np.ascontiguousarray(image_array[:, ::-1])
                    coords[0::2] = image_array.shape[1] - coords[0::2]
                    flip_coordinates(coords)
                
                keypoints = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                augmented = self.transforms(image=image_array, keypoints=keypoints)
                
                image_array = augmented['image']

                keypoints = augmented['keypoints']

                aug_coords = []
                for x, y in keypoints:
                    aug_coords.extend([x, y])

                coords = np.array(aug_coords, dtype=np.float32)

            x_mult = self.image_size / orig_w
            y_mult = self.image_size / orig_h

            coords[0::2] *= x_mult 
            coords[1::2] *= y_mult

            if self.normalize_targets:

                coords[0::2] /= self.image_size
                coords[1::2] /= self.image_size

            coords_tensor = torch.from_numpy(np.array(coords, dtype=np.float32))

        image_array = cv2.resize(
            image_array, 
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
            )
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        

        if self.return_meta:
            if self.has_coords:
                return image_tensor, coords_tensor, image_name, (orig_w, orig_h)
            else:
                return image_tensor, image_name, (orig_w, orig_h)
        elif self.has_coords:
            return image_tensor, coords_tensor
        else:
            raise RuntimeError('Dataset without coordinates not supported here')
