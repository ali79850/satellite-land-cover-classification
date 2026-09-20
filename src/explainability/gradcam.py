"""
Grad-CAM implementation for ResNet18.

Grad-CAM highlights which regions of an input image most influenced a
specific class prediction, by weighting the final convolutional layer's
feature maps with gradients flowing back from that class's output score.

Note: this is an approximate visualization of influential image regions,
not proof of model "reasoning" — see Phase 12 discussion in the README.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: shape (1, 3, H, W), already preprocessed
        target_class: class index to explain. If None, uses the model's
                      own top prediction.
        Returns: (heatmap as HxW numpy array in [0,1], predicted_class, confidence)
        """
        self.model.eval()
        output = self.model(input_tensor)  # (1, num_classes)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()
        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only positive influence, per Grad-CAM's original formulation

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()  # normalize to [0, 1]

        return cam, target_class, confidence


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    image_np: HxWx3, float in [0,1]
    heatmap: hxw (smaller, from the conv layer's spatial size), float in [0,1]
    Returns: HxWx3 overlay, float in [0,1]
    """
    import cv2
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * image_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)