from abc import abstractmethod
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from .base import BaseModel
import logging
from typing import Optional, Literal

MODE = Literal["train", "eval"]

"""
class BERT_BiLSTM_CRF(BaseModel):
    def __init__(self, num_classes: int, bert_model_name: str, device: str, hidden_size: int, num_layers: int):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=True,
                            batch_first=True).to(device)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_classes).to(device)
        self.crf = CRF(num_classes, batch_first=True).to(device)

    def forward(self, x, y=None):
        outputs = self.bert(**x)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        if y is not None:
            mask = x['attention_mask'].bool()
            loss = -self.crf(logits, y, mask=mask)
            return loss
        else:
            mask = x['attention_mask'].bool()
            tags = self.crf.decode(logits, mask=mask)
            return tags

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
"""


class BERT_BiLSTM_CRF(BaseModel):
    def __init__(self, num_classes: int, bert_model_name: str, device: str,
                 hidden_dim: Optional[int] = 256, num_layers: Optional[int] = 2):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True).to(device)
        self.dropout = nn.Dropout(p=0.1)
        self.hidden2label = nn.Linear(hidden_dim * 2, num_classes).to(device)
        self.crf = CRF(num_classes, batch_first=True).to(device)

    def forward(self, x):
        outputs = self.bert(**x)
        sequence_output = outputs.last_hidden_state
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2label(sequence_output)
        return emissions

    def loss(self, x, y_pred, y) -> torch.Tensor:
        mask = x['attention_mask'].bool()

        logging.debug(mask)

        log_likelihood = self.crf(y_pred, y, mask=mask)
        loss = -log_likelihood

        logging.debug(loss)
        logging.debug(f"loss.shape: {loss.shape}")

        loss_item = loss.item()
        loss.backward()
        return loss_item

    def decode(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        mask = mask.bool()
        return self.crf.decode(emissions, mask=mask)

    def predict(self, x):
        y_pred = self.forward(x)
        mask = x['attention_mask'].bool()
        logging.debug(y_pred.shape)
        y_pred = self.crf.decode(y_pred, mask=mask)
        logging.debug(y_pred)
        logging.debug(len(y_pred))
        # assert False
        return y_pred

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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def train_model(self, x, y):
        y_pred = self.forward(x)
        loss_item = self.loss(x, y_pred, y)
        return loss_item