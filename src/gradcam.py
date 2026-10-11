import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._save_activation)
        self.h2 = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=1):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam-cam.min())/(cam.max()-cam.min())
        return cam, logits.detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()

def resize_cam(cam, width, height):
    return cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

def overlay_cam(gray01, cam01):
    base = cv2.cvtColor(np.clip(gray01*255,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(np.clip(cam01*255,0,255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)

def peak_xy(cam01):
    y, x = np.unravel_index(np.argmax(cam01), cam01.shape)
    return int(x), int(y)
