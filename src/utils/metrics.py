import numpy as np


def mse(y_gt, y_pred):

    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)

    return np.mean((y_gt - y_pred) ** 2)


def rmse(y_gt, y_pred):
    
    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_gt - y_pred) ** 2))


def mae(y_gt, y_pred):

    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_gt - y_pred))


def nme(y_gt, y_pred, l=4, r=9):

    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)

    if y_gt.ndim == 1:
        y_gt = y_gt[None, :]
        y_pred = y_pred[None, :]

    y_gt = y_gt.reshape(y_gt.shape[0], -1, 2)
    y_pred = y_pred.reshape(y_pred.shape[0], -1, 2)

    left = y_gt[:, l]
    right = y_gt[:, r]

    eps = 1e-8
    d = np.linalg.norm(right - left, axis=1) + eps
    
    error = np.linalg.norm(y_gt - y_pred, axis=2).mean(axis=1)

    return np.mean(error / d)



def calculate_metric(
        keypoints_gt: list | np.ndarray, 
        keypoints_pred: list | np.ndarray, 
        metric: str | list[str] = 'mse'
        ) -> dict:
    
    """
    Calculates a metric between ground truth and predicted keypoint coordinates.

    Args:
        keypoints_gt (list | np.ndarray): Ground truth keypoint coordinates.
        keypoints_pred (list | np.ndarray): Predicted keypoint coordinates.
        metric (str | list[str]): Metric(or metrics) to compute. Supported values: "mse", "rmse", "mae", "nme".

    Returns:
        dict[str: float]: Pair of calculated metric(-s) name(-s) and value(-s).

    Raises:
        ValueError: If an unsupported metric is specified.
    """
    
    keypoints_gt = np.asarray(keypoints_gt)
    keypoints_pred = np.asarray(keypoints_pred)

    result = {}

    metrics = [metric] if isinstance(metric, str) else metric
    metrics = set(map(lambda x: x.lower(), metrics))

    supported_metrics = {'mse', 'rmse', 'mae', 'nme'}

    if len(metrics - supported_metrics) > 0:
        raise ValueError(f'Unknown metrics: {metrics - supported_metrics}')

    if 'mse' in metrics:
        
        result['mse'] = mse(keypoints_gt, keypoints_pred)
    
    if 'rmse' in metrics:

        result['rmse'] = rmse(keypoints_gt, keypoints_pred)
    
    if 'mae' in metrics:

        result['mae'] = mae(keypoints_gt, keypoints_pred)

    if 'nme' in metrics:

        result['nme'] = nme(keypoints_gt, keypoints_pred)

    return result