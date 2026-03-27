from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from src.utils.io import read_json
from src.utils.image import alpha_blend
from src.utils.keypoints import keypoint_name_to_idx_dict, complex_keypoints_to_coords_mapping

class FaceMask():

    def __init__(
            self,
            mask_name: str
            ):
        
        self.name = mask_name
        
        self.data_path = Path('./graphics/').resolve() / mask_name

        if not self.data_path.exists():
            raise ValueError(f'Could nor find mask: "{mask_name}" at {self.data_path}.')
        
        try:
            with Image.open(self.data_path / 'image.png') as image:
                self.image = image.convert('RGBA')
                self.image.load()

        except Exception as e:
            raise RuntimeError(f'Error trying to load image for mask "{mask_name}", check if \
                               it is a valid mask or try another mask.') from e
        
        try:
            keypoint_json = read_json(self.data_path / 'keypoints.json')

        except Exception as e:
            raise RuntimeError(f'Error trying to load keypoints for mask "{mask_name}", check if \
                               it is a valid mask or try another mask.') from e
        
        self.transform = keypoint_json['transform']
        self.keypoints = keypoint_json['keypoints']

        self._name2idx = keypoint_name_to_idx_dict

        self._complex2coords = complex_keypoints_to_coords_mapping

        for kp in self.keypoints:
            if (kp not in self._name2idx) and (kp not in self._complex2coords):
                raise ValueError(f'Unknown keypoint: "{kp}".')


    def _add_third_point(
        self,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:


        if len(points) < 2:
            raise ValueError('At least 2 points have to be given to add third.')
        
        elif len(points) >= 3:
            return points

        elif len(points) == 2:

            p1, p2 = points

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            p3 = (p1[0] - dy, p1[1] + dx)

            return [p1, p2, p3]
        

    def _calculate_keypoints(
        self,
        keypoints: list[float]
    ) -> dict:

        input_keypoints = {}

        for kp in self.keypoints:

            if kp in self._name2idx:
                idx = self._name2idx[kp]
                input_keypoints[kp] = (keypoints[idx[0]], keypoints[idx[1]])

            elif kp in self._complex2coords:
                res = self._complex2coords[kp](keypoints)
                input_keypoints[kp] = res

        return input_keypoints


    def _compute_transforms(
        self,
        image_keypoints: list,
    ):

        if self.transform == 'affine':

            mask_points_list = [self.keypoints[kp] for kp in self.keypoints]
            input_points_list = [image_keypoints[kp] for kp in image_keypoints]

            dst = np.array(self._add_third_point(input_points_list), dtype=np.float32)
            src = np.array(self._add_third_point(mask_points_list), dtype=np.float32)

            M = cv2.getAffineTransform(src, dst)

            return M


    def _warp(
            self, 
            M: cv2.Mat, 
            image_shape: tuple
            ):

        mask_np = np.asarray(self.image)

        mask_rgb = mask_np[..., :3]
        mask_alpha = mask_np[..., 3] / 255.0

        h, w = image_shape[:2]

        warped_rgb = cv2.warpAffine(
            src=mask_rgb, 
            M=M, 
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
            )
        
        warped_alpha = cv2.warpAffine(
            src=mask_alpha, 
            M=M, 
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
            )

        return warped_rgb, warped_alpha


    def apply(
            self,
            image: Image.Image | np.ndarray,
            keypoints: list[float]
    ) -> np.array:
        
        input_keypoints = self._calculate_keypoints(keypoints)

        transform_matrix = self._compute_transforms(input_keypoints)

        image_shape = np.asarray(image).shape[:2]
        transformed_mask, transformed_mask_alpha = self._warp(transform_matrix, image_shape)

        result = alpha_blend(
            image=image,
            mask=transformed_mask,
            alpha_map=transformed_mask_alpha
        )

        return result
