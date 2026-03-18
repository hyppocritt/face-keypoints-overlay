import os
import copy
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch 
from torch import nn 
from torch.utils.data import DataLoader

from src.dataset import FacePointsTransformDataset
from src.models.base_model import FacePointsModel
from src.utils.common import get_device, set_seed
from src.utils.io import load_from_state_dict
from src.utils.settings import Settings


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str | torch.device = None,
        epochs: int = 20,
        lr: float = 1e-3,
        use_scheduler: bool = False,
        weight_decay: float = 1e-5,
        early_stopping_delta: float = 1e-6,
        normalize_targets: bool = False,
        use_amp: bool = False,
        save_path: str | Path = './weights/',
        filename: str = 'latest',
        save_weights_only: bool = True,
        patience: int = 10,
):
    
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
        normalize_targets (bool): Whether to normalize coordinates during preprocessing. 
                                 Will make model predict normalized coordinates as well.
        use_amp (bool): Whether to use automatic mixed precision during training.
                        Enable for faster training and reduced memory usage.
        save_path (str | Path): Base directory for saving model checkpoints.
        filename (str): Filename for trained models' state dictionary.
        save_weights_only (bool): Whether to save models' state dictionary only.
                                  Otherwise the output will include epoch number and best validation loss.
        patience (int): Number of consecutive epochs without sufficient validation loss improvement before early stopping.

    Returns:
        nn.Module: Trained model (best weights if validation is used).
    """
    
    current_patience = patience
    save_path = Path(save_path) / (filename + '.pth')
    os.makedirs(save_path.parent, exist_ok=True)

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2,
            factor=0.5
        )

    best_val_mse = float('inf')

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss_sum = 0
        train_n = 0

        for images, coords in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images = images.to(device=device, dtype=torch.float32)
            coords = coords.to(device=device, dtype=torch.float32)

            if normalize_targets:
                coords /= float(images.size(-1))

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

        train_mse = train_loss_sum / max(1, train_n)

        if val_loader is not None:

            model.eval()

            val_loss_sum = 0
            val_n = 0

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for images, coords in val_loader:

                    images = images.to(device=device, dtype=torch.float32)
                    coords = coords.to(device=device, dtype=torch.float32)

                    if normalize_targets:
                        coords /= float(images.size(-1))

                    preds = model(images)
                    loss = criterion(preds, coords)

                    batch_size = images.size(0)
                    val_loss_sum += batch_size * loss.item()
                    val_n += batch_size

                val_mse = val_loss_sum / max(1, val_n)
                print(f'Epoch {epoch:02d}, train mse = {train_mse:.6f}   val mse = {val_mse:.6f}   val rmse = {val_mse ** 0.5:.4f}')

                if use_scheduler:
                    scheduler.step(val_mse)

                if val_mse >= best_val_mse - early_stopping_delta:
                    current_patience -= 1
                    if current_patience == 0:
                        print(f'No val MSE decrease, stopping training')
                        return model
                    
                elif save_path is not None:
                    
                    current_patience = patience
                    best_val_mse = val_mse
                    best_state_dict = copy.deepcopy(model.state_dict())

                    if save_weights_only:
                        torch.save(model.state_dict(), save_path)
                        print(f'Saved best model\'s state dict to {save_path}')

                    else:
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'val_mse': val_mse}, save_path)
                        print(f'Saved best model\'s checkpoint to {save_path}')

        else:
            print(f'Epoch {epoch:02d}, train mse = {train_mse:.6f}')

    model.load_state_dict(best_state_dict)
    return model

def get_train_val_loaders(
        dataset_dir: str | Path,
        metadata_path: str | Path,
        dataset_params: dict,
        dataloader_params: dict
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

    train_aug_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_no_aug_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    return train_aug_loader, val_no_aug_loader

def main(settings: Settings):

    set_seed()

    model_checkpoint_path = settings.resolve_model_path()
    dataset_dir = settings.data.path
    metadata_path = settings.data.metadata_path

    model = FacePointsModel(settings.model.input_size)
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
