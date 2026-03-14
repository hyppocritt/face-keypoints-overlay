import numpy as np


def calculate_metric(
        keypoints_gt: list | np.ndarray, 
        keypoints_pred: list | np.ndarray, 
        metric: str = 'mse'
        ) -> float:
    
    """
    Calculates a metric between ground truth and predicted keypoint coordinates.

    Args:
        keypoints_gt (list | np.ndarray): Ground truth keypoint coordinates.
        keypoints_pred (list | np.ndarray): Predicted keypoint coordinates.
        metric (str): Metric to compute. Supported values: "mse", "rmse".

    Returns:
        float: Calculated metric value.

    Raises:
        ValueError: If an unsupported metric is specified.
    """
    
    keypoints_gt = np.asarray(keypoints_gt)
    keypoints_pred = np.asarray(keypoints_pred)

    if metric == 'mse':
        
        return np.mean((keypoints_gt - keypoints_pred) ** 2)
    
    elif metric == 'rmse':

        return np.sqrt(np.mean((keypoints_gt - keypoints_pred) ** 2))
    
    else:
        raise ValueError(f'Unknown metric: {metric}')