import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import *


class BaseOptimizer:
    def __init__(self, optimizer_name: str, lr_scheduler: str, learning_rate: float, **kwargs):
        self.optimizer = None
        self.lr_scheduler = None
        if optimizer_name == "AdamW":
            self.optimizer = AdamW(kwargs["params"], learning_rate)
        else:
            raise ValueError(f"Invalid optimizer {optimizer_name}")

        if lr_scheduler == "Exp":
            self.lr_scheduler = ExponentialLR(self.optimizer, kwargs["gamma"])
        else:
            raise ValueError(f"Invalid lr_scheduler {lr_scheduler}")

