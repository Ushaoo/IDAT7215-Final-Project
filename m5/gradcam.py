import torch
from torch.nn import functional as F


class GradCAM:
    """Compute normalized Grad-CAM maps for a PyTorch classification model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._capture_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must output a BxCxHxW tensor")
        self.activations = output

    def _capture_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def generate(self, inputs, class_indices=None):
        if inputs.ndim != 4:
            raise ValueError("Expected inputs shaped BxCxHxW")
        inputs = inputs.detach().requires_grad_(True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(inputs)
            if logits.ndim != 2 or logits.shape[0] != inputs.shape[0]:
                raise ValueError("Model must return BxClasses logits")
            if class_indices is None:
                class_indices = logits.argmax(dim=1)
            elif isinstance(class_indices, int):
                class_indices = torch.full(
                    (inputs.shape[0],), class_indices, dtype=torch.long, device=inputs.device
                )
            else:
                class_indices = torch.as_tensor(class_indices, dtype=torch.long, device=inputs.device)
            if class_indices.shape != (inputs.shape[0],):
                raise ValueError("class_indices must contain one class per input")
            if class_indices.min() < 0 or class_indices.max() >= logits.shape[1]:
                raise ValueError("class_indices contains an out-of-range class")

            self.model.zero_grad(set_to_none=True)
            logits.gather(1, class_indices[:, None]).sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        maps = F.interpolate(maps, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        flat = maps.flatten(start_dim=1)
        minimum = flat.min(dim=1).values[:, None, None, None]
        maximum = flat.max(dim=1).values[:, None, None, None]
        denominator = maximum - minimum
        maps = torch.where(denominator > 1e-12, (maps - minimum) / denominator, torch.zeros_like(maps))
        return maps.detach(), logits.detach(), class_indices.detach()
