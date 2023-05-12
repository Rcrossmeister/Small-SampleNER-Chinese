from abc import abstractmethod

import torch
import torch.nn as nn
from transformers import BertModel
from .base import BaseModel
from typing import Literal
import logging

MODE = Literal["train", "eval"]


class BERT_Linear(BaseModel):
    def __init__(self, num_classes: int, bert_model_name: str, device: str):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes).to(device)

    def forward(self, x):
        outputs = self.bert(**x)
        logging.debug(outputs.last_hidden_state.shape)
        pooled_output = outputs.last_hidden_state
        logits = self.classifier(pooled_output)
        logging.debug(logits.shape)
        return logits

    def switch_mode(self, mode: MODE):
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        else:
            raise ValueError(f"Invalid mode {mode}")

    def to(self, x, y):
        x["input_ids"] = x["input_ids"].to(self.device)
        x["attention_mask"] = x["attention_mask"].to(self.device)
        return x, y.to(self.device)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        loss_fn = nn.CrossEntropyLoss()
        logging.debug(f"y_pred shape: {y_pred.shape}")
        y_pred = torch.transpose(y_pred, 1, 2)
        logging.debug(f"y_pred shape: {y_pred.shape}")
        loss = loss_fn(y_pred, y_true)
        loss_item = loss.item()
        loss.backward()
        return loss_item

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, **kwargs):
        pass

    def train_model(self, x, y) -> float:
        y_pred = self.forward(x)
        loss_item = self.loss(y_pred, y)
        return loss_item


class MultiGPU_BERT_Linear(BaseModel):
    def __init__(self, num_classes, bert_model_name='bert-base-uncased', device_ids=None):
        super().__init__()
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids
        self.model = BERT_Linear(num_classes=num_classes, bert_model_name=bert_model_name)
        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def switch_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f"Invalid mode {mode}")

    def to(self, x, y):
        pass

    def loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    def save_model(self, path):
        torch.save(self.model.module.state_dict(), path)

    def predict(self, **kwargs):
        pass

    @abstractmethod
    def training(self, **kwargs):
        pass


if __name__ == '__main__':
    logging.basicConfig(format="[%(levelname)s] %(filename)s - %(funcName)s, line %(lineno)d: %(message)s",
                        level=logging.DEBUG)
    test = BERT_Linear(20, "bert-base-chinese", "cuda:6")
    x = {'input_ids': torch.tensor([[ 101, 4906, 2825, 1059, 3175,  855, 6598, 6380, 3255, 5543, 8024, 2571,
         2949, 4638, 3749, 6756, 4495, 3833, 7444, 6206, 3300,  676, 2242,  671,
          756, 4263,  872,  102],
        [ 101, 2190, 8024, 6783, 5314,  671,  702, 1957,  782, 8024, 4638, 2768,
         5327,  511, 1927, 3307,  102,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0]])}

    y = torch.tensor([[ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
        [ 3,  3,  3,  3,  3,  3,  3, 11,  7,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3]])

    x, y = test.to(x, y)
    yp = test(x)
    loss = test.loss(yp, y)
    logging.debug(f"loss: {loss}")

