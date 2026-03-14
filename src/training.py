import os
import copy
import numpy as np 
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import torch 
from torch import nn 
from torch.utils.data import DataLoader

from src.dataset import FacePointsTransformDataset
from src.models.base_model import FacePointsModel
from src.utils.common import get_device, set_seed
from src.utils.io import load_from_state_dict, load_config
from src.paths import CONFIG_DIR, WEIGHTS_DIR


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = None,
        epochs: int = 20,
        lr: float = 1e-3,
        use_scheduler: bool = False,
        weight_decay: float = 1e-5,
        early_stopping_delta: float = 1e-6,
        normalize_targets: bool = False,
        use_amp: bool = False,
        save_path: str = './weights/',
        filename: str = 'latest',
        save_weights_only: bool = True,
        patience: int = 10,
):
    
    current_patience = patience
    save_path = Path(save_path) / (filename + '.pth')
    os.makedirs(save_path.parent, exist_ok=True)

    if device is None:
        device = get_device()
    
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
        metadata_path: str | Path,
        dataset_dir: str | Path,
        config: dict
        ):

    metadata = pd.read_csv(metadata_path)
    train_metadata, val_metadata = train_test_split(metadata, test_size=0.1, random_state=78)
    train_metadata.reset_index(drop=True, inplace=True)
    val_metadata.reset_index(drop=True, inplace=True)

    train_dataset = FacePointsTransformDataset(
        image_dir=dataset_dir,
        metadata_df=train_metadata,
        **config['dataset']
    )

    val_dataset = FacePointsTransformDataset(
        image_dir=dataset_dir,
        metadata_df=val_metadata,
        transforms=None,
        **config['dataset']
    )

    train_aug_loader = DataLoader(train_dataset, shuffle=True, **config['dataloader'])
    val_no_aug_loader = DataLoader(val_dataset, shuffle=False, **config['dataloader'])

    return train_aug_loader, val_no_aug_loader

def main():

    load_dotenv()
    set_seed()

    model_checkpoint_path = os.getenv('MODEL_CHECKPOINT_PATH', None)
    dataset_dir = os.getenv('IMAGE_DIR', None)
    metadata_path = os.getenv('METADATA_PATH', None)
    weights_dir = os.getenv('WEIGHTS_DIR', WEIGHTS_DIR)
    weights_dir = Path(weights_dir).resolve()

    config = load_config(CONFIG_DIR / 'train.yaml')

    model = FacePointsModel(**config['model'])
    if model_checkpoint_path is not None:
        load_from_state_dict(model, model_checkpoint_path)
        
    if dataset_dir is None or metadata_path is None:
        print('No dataset provided, training cancelled.')

    else:

        dataset_dir = Path(dataset_dir).resolve()
        metadata_path = Path(metadata_path).resolve()

        train_loader, val_loader = get_train_val_loaders(
            dataset_dir=dataset_dir,
            metadata_path=metadata_path,
            config=config
        )

        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            **config['training']
        )

if __name__ == '__main__':
    main()