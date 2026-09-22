import cv2 as cv
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import v2

class GradCAM:
    """
    A Class representing the GradCAM process to be used when generating the GradCAM heatmaps.

    Parameters
    ----------
        model: torch.nn.Module
            A trained ResNet-18 model to generate GradCAM heatmaps.

        target_layer: torch.nn.Module
            The layer in the model to target for the heatmaps.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, model:torch.nn.Module, input:torch.Tensor, output: torch.Tensor):
        """
        A helper function that registers the forward hook to capture the feature map activations.

        Parameters
        ----------
        model: torch.nn.Module
            A Res-Net-18 model to register the activations for.
        input: torch.Tensor
            The tensor input to the convolution layer of the hook.
        output: torch.Tensor
            The tensor output to the convolution layer of the hook.

        """
        self.activation = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad:torch.Tensor):
        """
        A helper function to capture the value of the gradients for the target layer.

        Parameters
        ----------
        grad: torch.Tensor
            The gradient for the target layer.

        """
        self.gradients=grad

    def generate(self, image:torch.Tensor, target_class:int | None=None):
        """
        Sends a single image through the model, and captures the gradients along with the final prediction.

        Parameters
        ----------
        image: torch.Tensor
            An image that has been transformed into a tensor acting as the model's input data

        target_class: int or None
            An int representing the target class of the image to be predicted. In most cases this value will be None,
            and we will use the models predicted class.


        Returns
        -------
        tuple: A tuple containing:
            - cam (torch.Tensor): A tensor representing a gradient heat map of the layer activations.
            - pred (int): An int representing the image's predicted class.

        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image)
        pred = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = pred

        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activation).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.detach().cpu(), pred

    def remove(self):
        """A helper function used to remove the forward hook after GradCAM is completed."""
        self.handle.remove()

def create_overlay(heatmap:torch.Tensor, display_image: Image.Image | np.ndarray, alpha: float|int=0.45):
    """
    Generates the original display image with the GradCAM heatmap overlay.

    Parameters
    ----------
    heatmap: torch.Tensor
        A tensor heatmap of the gradient feature map activations.

    display_image: Image.Image | np.ndarray
        A PIL Image containing the original input image

    alpha: float | int
        A float, or int, value representing the weighted transparency factor for the final image. See documentation for
        cv2.addWeighted for more information.

    Returns
    -------
    overlay: np.ndarray
        A numpy array representing a scaled heatmap over the original image passed through the model.

    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().numpy()

    if isinstance(display_image, Image.Image):
        display_image = np.array(display_image)

    resized = cv.resize(heatmap,(display_image.shape[1], display_image.shape[0]))

    heatmap_uint8 = np.uint8(255 * resized)
    color_heatmap = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)
    color_heatmap = cv.cvtColor(color_heatmap, cv.COLOR_BGR2RGB)

    overlay = cv.addWeighted(display_image, 1-alpha, color_heatmap, alpha, 0)

    return overlay

def prepare_gradcam_images(image_path:str | Path, model_transform:v2.Compose, device:torch.device):
    """
    Performs any pre-processing steps need for the GradCAM process.

    Parameters
    ----------
    image_path: str or Path
        A str or Path object pointing to the image to be processed.

    model_transform: torchvision.transforms.v2
        A v2 composition transform used to test images on the fully-trained model.

    device: torch.device
        A torch.device used to process the GradCAM

    Returns
    -------

    display_image: PIL.Image.Image
        A PIL Image containing the original input image after the display transformations.

    model_image: torch.Tensor
        A tensor containing the original input image after the model transformations. After this method is called, the
        model image has already been sent to the device (provided that the device parameter is not None) and is
        waiting for the GradCAM process.
    """
    image = Image.open(image_path).convert("RGB")

    display_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224)
    ])

    display_image = display_transform(image)
    model_image = model_transform(image)

    model_image = model_image.unsqueeze(0).to(device)

    return display_image, model_image

