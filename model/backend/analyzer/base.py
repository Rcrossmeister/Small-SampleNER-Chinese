from abc import ABCMeta, abstractmethod


class BaseAnalyzer(metaclass=ABCMeta):
    @abstractmethod
    def update_loss(self, **kwargs) -> None:
        pass

    @abstractmethod
    def update_eval_value(self, x, y_pred, y_true) -> None:
        pass

    @abstractmethod
    def update_epoch(self) -> None:
        pass

    @abstractmethod
    def print_loss(self):
        pass

    @abstractmethod
    def print_eval_value(self):
        pass




