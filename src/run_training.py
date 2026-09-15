import numpy as np

from pathlib import Path
import pandas as pd

from sklearn.utils.class_weight import  compute_class_weight

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.dataset import load_metadata, create_splits, create_transformations, CassavaDataset
from src.model import create_model
from src.train import train_model

def define_loss_optim(df: pd.DataFrame,
                      model,
                      device: torch.device | str,
                      layers:str|list[str] = ['layer4', 'fc'],
                      lrs: float|int|list[float|int] = [1e-4, 1e-3],
                      loss_fn:type[nn.Module] = nn.CrossEntropyLoss,
                      optimizer = torch.optim.Adam,
                      loss_weights:str|None='unweighted',
                      loss_kwargs:dict | None = None,
                      optimizer_kwargs:dict | None = None):
    """

    """

    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    if not isinstance(loss_fn, type) or not issubclass(loss_fn, nn.Module):
        raise TypeError(f"loss_fn must be an nn.Module, got {loss_fn!r}")

    if isinstance(loss_weights, str) or loss_weights is None:
        match loss_weights:
            case 'balanced':
                class_weights = compute_class_weight(class_weight='balanced',
                                                     classes=np.sort(df["label"].unique()),
                                                     y=df["label"])
                class_weights =torch.tensor(class_weights, dtype=torch.float32, device=device)

            case 'unweighted' | None:
                class_weights = None

            case 'sqrt':
                class_count = df["label"].value_counts().sort_index()
                counts = torch.tensor(class_count.values,
                                      dtype=torch.float32)
                class_weights = 1.0 / torch.sqrt(counts)
                class_weights = class_weights / class_weights.mean()
                class_weights = class_weights.to(device)

            case _:
                raise ValueError(f"Unknown class weight parameter: {loss_weights}\n"
                                 f"Please choose from balanced, unweighted, or sqrt.")

    else:
        try:
            class_weights = torch.as_tensor(loss_weights, dtype=torch.float32, device=device)

        except (TypeError, ValueError) as err:
            raise TypeError("loss weights must either be a named weighting scheme or "
                            "an array-like collection of numeric class weights.") from err

        n_classes = df["label"].nunique()

        if class_weights.ndim != 1:
            raise ValueError(f"Class weights must be 1-dimensional, got shape {tuple(class_weights.shape)}")

        if class_weights.numel() != n_classes:
            raise ValueError(f"Expected {n_classes} class weights, got {class_weights.numel()}")

        if not torch.isfinite(class_weights).all():
            raise ValueError("Loss weights must contain only finite values.")

        if (class_weights < 0).any():
            raise ValueError("Loss weights must contain only non-negative values.")

    if class_weights is not None:
        if "weight" in loss_kwargs:
            raise ValueError("Specify class weights using either loss_weights or loss_kwargs['weight'], not both.")

        loss_kwargs['weight'] = class_weights

    try:
        criterion = loss_fn(**loss_kwargs)
    except TypeError as err:
        raise TypeError(f"Failed to initialize {loss_fn.__name__} with the supplied loss configuration.") from err

    if not isinstance(optimizer, type) or not issubclass(optimizer, torch.optim.Optimizer):
        raise TypeError(f"Optimizer must be an torch.optim.Optimizer, got {optimizer!r}")

    if isinstance(layers, str):
        layers = [layers]

    if isinstance(lrs, (int, float)):
        lrs = [lrs]

    if len(layers) != len(lrs):
        raise ValueError(f"Number of layers and number of learning rates do not match.")

    optimizer_params = []
    for lr, layer_name in zip(lrs, layers):
        try:
            layer = model.get_submodule(layer_name)
        except AttributeError as err:
            available_layers = [layer_name for layer_name, _ in model.named_children()]
            raise ValueError(
                f"Layer {layer_name!r} not available for ResNet-18\n"
                f"Available layers: {', '.join(available_layers)}"
            ) from err

        optimizer_params.append({'params': layer.parameters(), 'lr': lr})

    try:
        optim = optimizer(optimizer_params, **optimizer_kwargs)
    except TypeError as err:
        raise TypeError(f"Failed to initialize {optimizer.__name__} with the supplied optimizer configuration.") \
            from err

    return criterion,optim

def history_to_df(history, stage):
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
                       checkpoint_name='best_finetuned_resnet18.pt', image_directory=image_dir,
                       reporting=True, warm_cache=False)

    frozen_df = history_to_df(frozen_data, "frozen")
    fintuned_df = history_to_df(finetuned_data, "finetuned")

    history_df = pd.concat([frozen_df, fintuned_df], ignore_index=True)
    history_df.to_csv("results/training_history.csv", index=False)

if __name__ == '__main__':
    main()
