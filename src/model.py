import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet
import torch.nn as nn

def create_model(unfreeze_layers: str | list[str],
                    checkpoint=None,
                    num_classes:int = 5) -> ResNet:
    """

    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if checkpoint:
        checkpoint = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    layers_to_unfreeze = ([unfreeze_layers] if isinstance(unfreeze_layers, str) else unfreeze_layers)

    for layer_name in layers_to_unfreeze:
        try:
           layer = model.get_submodule(layer_name)
        except AttributeError as exc:
            available_layers = [layer for layer, _ in model.named_children()]
            raise ValueError(
                f"Layer {layer_name!r} not available for ResNet-18\n"
                f"Available layers: {', '.join(available_layers)}"
            ) from exc

        for param in layer.parameters():
            param.requires_grad = True

    return model