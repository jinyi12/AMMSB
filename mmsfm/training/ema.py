
import torch
import torch.nn as nn

class EMA:
    """Exponential Moving Average for model parameters.

    Maintains a shadow copy of model parameters that are updated as an
    exponential moving average. This can stabilize training and improve
    generalization, especially for fixed-step ODE solvers.

    Args:
        model: The model to track.
        decay: EMA decay rate (typically 0.999 or 0.9999).
        device: Device for shadow parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu"):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        self._register()

    def _register(self):
        """Initialize shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Track parameters that are trainable at construction time.
                # Note: policies may later be temporarily frozen/unfrozen; we still
                # want EMA to apply/restore for tracked parameters regardless of
                # their current `requires_grad` flag.
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model parameters with shadow parameters."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data = self.backup[name].clone()
        self.backup = {}
