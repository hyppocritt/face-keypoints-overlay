import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch 
from torch import nn 
from torch.utils.data import DataLoader

from src.dataset import FacePointsTransformDataset
from src.models.base_model import FacePointsModel
from src.models.resnet_like import FacePointsResNet
from src.utils.common import get_device, set_seed
from src.utils.io import load_from_state_dict
from src.utils.settings import Settings
from src.utils.metrics_torch import mse_torch, mae_torch, nme_torch


class WingLoss(nn.Module):

    def __init__(self, w=10, epsilon=2):

        super().__init__()

        self.w = w
        self.epsilon = epsilon
        self.c = w - w * np.log(1 + w / epsilon)

    def forward(self, y_gt, y_pred):

        e = torch.abs(y_gt - y_pred)

        small = e < self.w
        big = ~small

        loss = torch.zeros_like(e)

        loss[small] = self.w * torch.log(1 + e[small] / self.epsilon)
        loss[big] = e[big] - self.c

        return loss.mean()
    

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str | torch.device = None,
        epochs: int = 20,
        lr: float = 1e-3,
        loss: str = 'mse',
        use_scheduler: bool = False,
        weight_decay: float = 1e-5,
        early_stopping_delta: float = 1e-6,
        use_amp: bool = False,
        save_dir: str | Path = './weights/',
        filename: str = 'latest',
        save_weights_only: bool = True,
        patience: int = 10,
        return_metrics: bool = False
    ) -> nn.Module | tuple[nn.Module, dict]:
    
    """
    Runs full training pipeline with specified parameters.

    Args:
        model (nn.Module): PyTorch model object for face keypoint regression.
        train_loader (DataLoader): PyTorch Dataloader for train split.
        val_loader (DataLoader | None): PyTorch Dataloader for val split.
                                        If None, validation is skipped.
        device (str | torch.device): Device type or PyTorch Device to do calculations on.
        epochs (int): Amount of training epochs. 
                      Training may be stopped earlier if early stopping is triggered.
        lr (float): Learning rate used by the optimizer.
        use_scheduler (bool): Wether to use learning rate scheduler that decreases lr of plateau.
                              Enables learning rate scheduling on validation plateau, which may improve convergence.
        weight_decay (float): Weight decay for AdamW optimizer.
        early_stopping_delta (float): Minimum improvement in validation loss required to reset early stopping.
        use_amp (bool): Whether to use automatic mixed precision during training.
                        Enable for faster training and reduced memory usage.
        save_dir (str | Path): Base directory for saving model checkpoints.
        filename (str): Filename for trained models' state dictionary.
        save_weights_only (bool): Whether to save models' state dictionary only.
                                  Otherwise the output will include epoch number and best validation loss.
        patience (int): Number of consecutive epochs without sufficient validation loss improvement before early stopping.
        return_metrics (bool): Whether to return metrics of the best model. 
                               Changes output from nn.Module to tuple[nn.Module, dict].
    Returns:
        nn.Module | tuple[nn.Module, dict]: Trained model (best weights if validation is used) 
                    or tuple with trained model and its metrics.

    """
    
    current_patience = patience
    save_dir = Path(save_dir) / (filename + '.pth')
    os.makedirs(save_dir.parent, exist_ok=True)

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    model.to(device)

    if loss.lower() not in {'mse', 'smoothl1', 'wing'}:
        raise ValueError(f'Unknown loss: {loss}. Use "mse", "smoothl1" or "wing".')
    
    if loss.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss.lower() == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    elif loss.lower() == 'wing':
        criterion = WingLoss()

    loss_name = loss
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2,
            factor=0.5
        )

    best_val_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())
    metrics = {}

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss_sum = 0

        train_mse_sum = 0
        train_mae_sum = 0
        train_nme_sum = 0

        train_n = 0

        for images, coords in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images = images.to(device=device, dtype=torch.float32)
            coords = coords.to(device=device, dtype=torch.float32)

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, coords)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            train_loss_sum += batch_size * loss.item()
            train_n += batch_size

            with torch.no_grad():

                train_mse_sum += batch_size * mse_torch(coords, preds).item()
                train_mae_sum += batch_size * mae_torch(coords, preds).item()
                train_nme_sum += batch_size * nme_torch(coords, preds).item()

        train_loss = train_loss_sum / max(1, train_n)

        train_mse = train_mse_sum / max(1, train_n)
        train_mae = train_mae_sum / max(1, train_n)
        train_nme = train_nme_sum / max(1, train_n)

        if val_loader is not None:

            model.eval()

            val_loss_sum = 0

            val_mse_sum = 0
            val_mae_sum = 0
            val_nme_sum = 0

            val_n = 0

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for images, coords in val_loader:

                    images = images.to(device=device, dtype=torch.float32)
                    coords = coords.to(device=device, dtype=torch.float32)

                    preds = model(images)
                    loss = criterion(preds, coords)

                    batch_size = images.size(0)
                    val_loss_sum += batch_size * loss.item()
                    val_n += batch_size

                    val_mse_sum += batch_size * mse_torch(coords, preds).item()
                    val_mae_sum += batch_size * mae_torch(coords, preds).item()
                    val_nme_sum += batch_size * nme_torch(coords, preds).item()

                val_loss = val_loss_sum / max(1, val_n)

                val_mse = val_mse_sum / max(1, val_n)
                val_mae = val_mae_sum / max(1, val_n)
                val_nme = val_nme_sum / max(1, val_n)

                print(
                    f'''Epoch {epoch:02d}. Optimizing {loss_name.upper()}
                    train loss = {train_loss:.6f}   val loss = {val_loss:.6f}
                    train MSE = {train_mse:.6f}   val MSE = {val_mse:.6f}
                    train RMSE = {train_mse ** 0.5:.4f} val RMSE = {val_mse ** 0.5:.4f}
                    train MAE = {train_mae:.6f} val MAE = {val_mae:.6f}
                    train NME = {train_nme:.6f} val NME = {val_nme:.6f}
                    ''')

                if use_scheduler:
                    scheduler.step(val_loss)

                if val_loss >= best_val_loss - early_stopping_delta:
                    current_patience -= 1
                    if current_patience == 0:
                        print(f'No val MSE decrease, stopping training')

                        model.load_state_dict(best_state_dict)

                        if return_metrics:
                            return (model, metrics)
                        else:
                            return model
                    
                else:
                    
                    current_patience = patience
                    best_val_loss = val_loss

                    best_val_mse = val_mse
                    best_val_rmse = val_mse ** 0.5
                    best_val_mae = val_mae
                    best_val_nme = val_nme

                    metrics = {
                        'mse': best_val_mse,
                        'rmse': best_val_rmse,
                        'mae': best_val_mae,
                        'nme': best_val_nme
                    }

                    if save_dir is not None:
                        best_state_dict = copy.deepcopy(model.state_dict())

                        if save_weights_only:
                            torch.save(model.state_dict(), save_dir)
                            print(f'Saved best model\'s state dict to {save_dir}')

                        else:
                            torch.save({'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'epoch': epoch,
                                        'metrics': metrics}, save_dir)
                            print(f'Saved best model\'s checkpoint to {save_dir}')

        else:
            print(f'''Epoch {epoch:02d}. Optimizing {loss_name.upper()}
                    train loss = {train_loss:.6f}
                    train MSE = {train_mse:.6f}   train RMSE = {train_mse ** 0.5:.4f} 
                    train MAE = {train_mae:.6f} train NME = {train_nme:.6f}
                  ''')

    model.load_state_dict(best_state_dict)

    if return_metrics:
        return (model, metrics)
    
    else:
        return model
    

def get_train_val_loaders(
        dataset_dir: str | Path,
        metadata_path: str | Path,
        dataset_params: dict = None,
        dataloader_params: dict = None
        ) -> tuple[DataLoader, DataLoader]:
    
    """
    Prepares dataloaders for training.

    Args:
        dataset_dir (str | Path): Directory containing dataset.
        metadata_path (str | Path): Path to metadata file with ground truth.
        dataset_params (dict): Dictoinary with parameters used in dataset initialization.
        dataloader_params (dict): Dictoinary with parameters used in dataloader initialization.

    Returns:
        tuple[Dataloader, Dataloader]: Train and validation PyTorch dataloaders.
    """

    metadata = pd.read_csv(metadata_path)
    train_metadata, val_metadata = train_test_split(metadata, test_size=0.1, random_state=78)
    train_metadata.reset_index(drop=True, inplace=True)
    val_metadata.reset_index(drop=True, inplace=True)

    if dataset_params is None:
        dataset_params = {}

    train_dataset = FacePointsTransformDataset(
        image_dir=dataset_dir,
        metadata_df=train_metadata,
        **dataset_params
    )

    val_dataset = FacePointsTransformDataset(
        image_dir=dataset_dir,
        metadata_df=val_metadata,
        transforms=None,
        **dataset_params
    )

    if dataloader_params is None:
        dataloader_params = {}

    train_aug_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_no_aug_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    return train_aug_loader, val_no_aug_loader

def main(settings: Settings):

    set_seed()

    model_checkpoint_path = settings.resolve_model_path()
    dataset_dir = settings.data.path
    metadata_path = settings.data.metadata_path

    if settings.model.model_type.lower() == 'resnet':
        model = FacePointsResNet(settings.model.input_size)
    elif settings.model.model_type.lower() == 'base':
        model = FacePointsModel(settings.model.input_size)
    else: 
        raise ValueError(f'Unknown model type: {settings.model.model_type}. Use "base" or "resnet".')
    if model_checkpoint_path is not None:
        load_from_state_dict(model, model_checkpoint_path)
        
    if dataset_dir is None or metadata_path is None:
        raise ValueError('No dataset provided, training cancelled.')

    else:

        train_loader, val_loader = get_train_val_loaders(
            dataset_dir=dataset_dir,
            metadata_path=metadata_path,
            dataset_params=settings.dataset,
            dataloader_params=settings.dataloader
        )

        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            **settings.training
        )
