from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import load_metadata, create_splits, create_transformations, CassavaDataset
from model import create_model, define_loss_optim
from train import train_model


def history_to_df(history, stage):
    """


    Parameters
    ----------


    Returns
    -------

    """
    return pd.DataFrame({
        "stage": stage,
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_accuracy": history["train_accuracy"],
        "val_accuracy": history["val_accuracy"]})

def main():
    data_dir = Path('data')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, label_map, image_dir = load_metadata(data_dir)

    train_df, val_df, _ = create_splits(df)
    train_transform, val_transform, _ = create_transformations()

    train_dataset = CassavaDataset(train_df, image_dir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers = 4,
                             persistent_workers=True, prefetch_factor=4)

    val_dataset = CassavaDataset(val_df, image_dir, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers = 4,
                            persistent_workers=True, prefetch_factor=4)

    # Frozen backbone
    model = create_model(unfreeze_layers=['fc']).to(device)

    loss_fn, optimizer = define_loss_optim(train_df, model, device, layers='fc', lrs=1e-3)

    frozen_data = train_model(train_loader, val_loader, model, loss_fn, optimizer, device,
                       checkpoint_name='best_frozen_resnet18.pt', image_directory=image_dir,
                       reporting=True, warm_cache=True)

    # Partial fine_tuning
    model = create_model(unfreeze_layers=['layer4', 'fc'], checkpoint='best_frozen_resnet18.pt').to(device)
    loss_fn, optimizer = define_loss_optim(train_df, model, device, layers=['layer4', 'fc'], lrs=[1e-4, 1e-3])

    finetuned_data = train_model(train_loader, val_loader, model, loss_fn, optimizer, device,
                       checkpoint_name='TEST.pt', image_directory=image_dir,
                       reporting=True, warm_cache=False)

    frozen_df = history_to_df(frozen_data, "frozen")
    fintuned_df = history_to_df(finetuned_data, "finetuned")

    history_df = pd.concat([frozen_df, fintuned_df], ignore_index=True)
    history_df.to_csv("results/training_history.csv", index=False)

if __name__ == '__main__':
    main()
