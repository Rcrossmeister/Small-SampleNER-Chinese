import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from frontend.dataset.base import BaseDataset
from frontend.model.base import BaseModel
from backend.optimizer.base import BaseOptimizer
from backend.analyzer.base import BaseAnalyzer


class Pipeline:
    def __init__(self, train_dataloader: DataLoader, eval_dataloader: DataLoader,
                 model: BaseModel, optimizer: BaseOptimizer, analyzer: BaseAnalyzer,
                 epochs: int, log_level: str,
                 **kwargs):
        # logging.basicConfig(format="[%(levelname)s] %(filename)s - %(funcName)s, line %(lineno)d: %(message)s",
        #                     level=log_level)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.optimizer = optimizer
        self.analyzer = analyzer
        self.epochs = epochs
        self.kwargs = kwargs

    def train(self):
        for epoch in range(self.epochs):
            logging.info(f"epoch: {epoch + 1}")
            self.model.switch_mode("train")
            for x, y in self.train_dataloader:

                self.optimizer.optimizer.zero_grad()

                x, y = self.model.to(x, y)
                # y_pred = self.model(x)
                # loss = self.model.loss(x, y_pred, y)
                loss = self.model.train_model(x, y)
                logging.info(f"loss: {loss}")
                self.analyzer.update_loss(loss)

                self.optimizer.optimizer.step()
            self.optimizer.lr_scheduler.step()
            # last_loss, avg_loss, epoch_avg_loss = self.analyzer.get_loss()
            # logging.info(f"train/avg_loss: {avg_loss:.2f}, train/epoch_avg_loss: {epoch_avg_loss:.2f}")
            self.analyzer.print_loss()
            self.eval()
            self.analyzer.update_epoch()
            logging.info(f"lr: {self.optimizer.lr_scheduler.get_last_lr()}")

    def eval(self):
        self.model.switch_mode("eval")
        with torch.no_grad():
            for x, y in self.eval_dataloader:
                x, y = self.model.to(x, y)
                y_pred = self.model.predict(x)
                logging.debug(y.shape)

                self.analyzer.update_eval_value(x, y_pred, y)
        self.analyzer.print_eval_value()

    def test(self):
        pass

