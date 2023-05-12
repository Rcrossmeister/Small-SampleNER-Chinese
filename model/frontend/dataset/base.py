from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod


class BaseDataset(Dataset, metaclass=ABCMeta):

    @abstractmethod
    def load_data(self, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

