import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import  compute_class_weight
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def define_loss_optim(label_df: pd.DataFrame,
                      model,
                      device: torch.device ,
                      layers:str|tuple = ['layer4', 'fc'],
                      lrs: float|int|tuple = [1e-4, 1e-3],
                      loss_fn:type[nn.Module] = nn.CrossEntropyLoss,
                      optimizer:type[torch.optim.Optimizer] = torch.optim.Adam,
                      loss_weights:str|np.ndarray|torch.Tensor='unweighted',
                      loss_kwargs:dict | None = None,
                      optimizer_kwargs:dict | None = None):
    """
    Creates the loss function and optimizer for the ResNet-18 model.

    Parameters
    ----------
        label_df : pd.DataFrame
            A DataFrame containing the image names and label IDs for the images in the training dataset.

        model: torch.nn.Module
            A torch.nn.Module representing the ResNet-18 model.

        device: torch.device or str
            A torch.device representing the device on which the model is to be trained.

        layers: str or tuple
            A string or list of strings containing which layers of the to adjust optimizer's learning rates for

        lrs: float, int, or tuple
            A float, int, or list of floats/ints corresponding to the learning rates of each layer. If a list of
            learning rates is passed, the method assumes that they are in the same order as the layers provide. For
            example, if our layers parameter is ['layer4', 'fc'] and lrs parameter is [1e-4, 1e-3], the method assumes
            layer4 has a lr of 1e-4 and the fc layer has a lr of 1e-3.

        loss_fn: torch.nn.Module
            A nn.Module loss function. This method allows custom loss function and checks some basic conditions.
            However, consistent and correct functionality of custom methods is assumed.

        optimizer: torch.optim.Optimizer
            A torch.optim optimizer. As with loss functions, this methods allows for custom optimizer but overall
            correctness and consistent functionality is assumed.

        loss_weights: str, np.nd, or torch.Tensor
            A string for a pre-defined weight selection or array-like collection of weights for the loss function. If
            the parameter is a string, the method expects either 'balanced', 'unweighted', or 'sqrt' for a balanced
            (determined by sklearn.utils.class_weight.compute_class_weight), unweighted, or square-root class weighting
            respectively.

        loss_kwargs: dict or None
            If not None, a dictionary containing the argument name and respective value for addition loss function
            parameters. For example, one could pass {'label_smoothing': 0.1} to allow the label_smoothing parameter of
            a loss function to be set to 0.1

        optimizer_kwargs: dict or None
            If not None, a dictionary containing the argument name and respective value for addition optimizer
            parameters.

    Returns
    -------
        tuple: A tuple containing:
            criterion (nn.Module): a nn.Module loss function with any weights or keywords set.
            optim (torch.optim.Optimizer): a torch.optim optimizer with all learning rates and any keywords set.

    """

    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    if not isinstance(loss_fn, type) or not issubclass(loss_fn, nn.Module):
        raise TypeError(f"loss_fn must be an nn.Module, got {loss_fn!r}")

    if isinstance(loss_weights, str) or loss_weights is None:
        match loss_weights:
            case 'balanced':
                class_weights = compute_class_weight(class_weight='balanced',
                                                     classes=np.sort(label_df["label"].unique()),
                                                     y=label_df["label"])
                class_weights =torch.tensor(class_weights, dtype=torch.float32, device=device)

            case 'unweighted' | None:
                class_weights = None

            case 'sqrt':
                class_count = label_df["label"].value_counts().sort_index()
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

        n_classes = label_df["label"].nunique()

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

    return criterion, optim

def create_model(unfreeze_layers: str | tuple = ['layer4', 'fc'],
                    checkpoint: str| None | Path =None,
                    num_classes:int = 5, checkpoint_weights_only:bool = False):
    """
    Creates a ResNet-18 model with any specified layers unfrozen. This method is also used to load any checkpoints and
    ensure the output layer matches the correct number of classes.

    Parameters
    ----------
        unfreeze_layers: str or tuple
            A string or list of strings containing the name of layers to unfreeze.

        checkpoint: str, Path, or None
            If not None, a string containing the path to any pt file containing any pre-trained model.

        num_classes: int
            The number of classes to predict.

        checkpoint_weights_only: bool
            If True, the passed checkpoints contains only weights. If you are using a checkpoint generated from this
            project, this condition should be sent to False.

    Returns
    -------
        model: ResNet
            A ResNet-18 model with any specified layers unfrozen.

    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if checkpoint:
        if checkpoint_weights_only == False:
            checkpoint = torch.load(checkpoint, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    layers_to_unfreeze = ([unfreeze_layers] if isinstance(unfreeze_layers, str) else unfreeze_layers)

    for layer_name in layers_to_unfreeze:
        try:
           layer = model.get_submodule(layer_name)
        except AttributeError as err:
            available_layers = [layer for layer, _ in model.named_children()]
            raise ValueError(
                f"Layer {layer_name!r} not found in model\n"
                f"Available layers: {', '.join(available_layers)}"
            ) from err

        for param in layer.parameters():
            param.requires_grad = True

    return model