import cv2
import cv2 as cv
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activation = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients=grad

    def generate(self, image, target_class=None):
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
        self.handle.remove()

def create_overlay(heatmap, display_image, alpha=0.45):
    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().numpy()

    if isinstance(display_image, Image.Image):
        display_image = np.array(display_image)

    resized = cv.resize(heatmap,
                        (display_image.shape[1], display_image.shape[0]))

    heatmap_uint8 = np.uint8(255 * resized)
    color_heatmap = cv.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv.cvtColor(color_heatmap, cv.COLOR_BGR2RGB)

    overlay = cv.addWeighted(display_image, 1-alpha, color_heatmap, alpha, 0)

    return overlay

def prepare_gradcam_images(image_path, model_transform, device=None):
    image = Image.open(image_path).convert("RGB")

    display_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224)
    ])

    display_image = display_transform(image)
    model_image = model_transform(image)

    if device is not None:
        model_image = model_image.unsqueeze(0).to(device)

    return display_image, model_image

