import torch


def mse_torch(y_gt, y_pred):

    return torch.mean((y_gt - y_pred) ** 2)


def rmse_torch(y_gt, y_pred):

    return torch.sqrt(torch.mean((y_gt - y_pred) ** 2))


def mae_torch(y_gt, y_pred):

    return torch.mean(torch.abs(y_gt - y_pred))


def nme_torch(y_gt, y_pred, l=4, r=9):

    if y_gt.ndim == 1:
        y_gt = y_gt.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)

    y_gt = y_gt.reshape(y_gt.shape[0], -1, 2)
    y_pred = y_pred.reshape(y_pred.shape[0], -1, 2)

    left = y_gt[:, l]
    right = y_gt[:, r]

    eps = 1e-8
    d = torch.norm(right - left, dim=1) + eps

    error = torch.norm(y_gt - y_pred, dim=2).mean(dim=1)

    return torch.mean(error / d)
