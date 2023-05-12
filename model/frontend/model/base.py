from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def switch_mode(self, **kwargs):
        pass

    @abstractmethod
    def to(self, x, y):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def save_model(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def train_model(self, **kwargs) -> float:
        pass

