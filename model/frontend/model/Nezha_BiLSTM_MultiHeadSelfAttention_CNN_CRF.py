import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, NezhaModel
from torchcrf import CRF
from .base import BaseModel
import logging
from typing import Optional, Literal, List
from .MultiHead_Attention import MultiHead_SelfAttention

MODE = Literal["train", "eval"]


class Nezha_BiLSTM_MultiHeadSelfAttention_CNN_CRF(BaseModel):
    def __init__(self, num_classes: int, bert_model_name: str, device: str,
                 lstm_hidden_size: Optional[int] = 256, lstm_num_layers: Optional[int] = 2, attention_head: Optional[int] = 8,
                 cnn_kernel_sizes: Optional[List[int]] = None, cnn_filter_sizes: Optional[List[int]] = None,
                 dropout_p: Optional[float] = 0.1):
        super().__init__()
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [2, 3, 4]
        if cnn_filter_sizes is None:
            cnn_filter_sizes = [128, 128, 128]
        self.device = device
        self.dropout_p = dropout_p

        # BERT model
        self.bert = NezhaModel.from_pretrained(bert_model_name).to(device)

        # BiLSTM model
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        ).to(device)

        self.attention = MultiHead_SelfAttention(lstm_hidden_size * 2, attention_head).to(device)

        # TextCNN model
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_filter_sizes = cnn_filter_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (k_size, self.bert.config.hidden_size)) for k_size, n_filters in zip(cnn_kernel_sizes, cnn_filter_sizes)
        ]).to(device)

        # CRF model
        self.num_classes = num_classes
        self.crf = CRF(num_tags=num_classes, batch_first=True).to(device)

        # Final classifier
        self.classifier = nn.Linear(sum(cnn_filter_sizes) + 2 * lstm_hidden_size, num_classes).to(device)

    def forward(self, x):
        # BERT model
        bert_outputs = self.bert(**x)

        # BiLSTM model
        lstm_outputs, _ = self.lstm(bert_outputs.last_hidden_state)   # [batch, seq len, hidden * 2]
        lstm_outputs = F.dropout(lstm_outputs, p=self.dropout_p, training=self.training)

        logging.debug(f"lstm shape: {lstm_outputs.shape}")

        lstm_outputs = self.attention(lstm_outputs)

        logging.debug(f"attention shape: {lstm_outputs.shape}")

        # TextCNN model
        cnn_inputs = bert_outputs.last_hidden_state.unsqueeze(1)
        logging.debug(f"cnn shape: {cnn_inputs.shape}")
        cnn_outputs = [F.relu(conv(cnn_inputs)).squeeze(3) for conv in self.convs]
        logging.debug(f"cnn shape: {cnn_outputs[0].shape}")
        cnn_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_outputs]
        logging.debug(f"cnn shape: {cnn_outputs[0].shape}")
        cnn_outputs = torch.cat(cnn_outputs, 1)
        logging.debug(f"cnn shape: {cnn_outputs.shape}")
        # logging.debug(f"cnn: {cnn_outputs}")
        cnn_outputs = F.dropout(cnn_outputs, p=self.dropout_p, training=self.training) # [batch, sum]

        # Combine BiLSTM and TextCNN outputs
        cnn_outputs = torch.unsqueeze(cnn_outputs, dim=1).repeat(1, lstm_outputs.shape[1], 1)
        logging.debug(f"cnn shape: {cnn_outputs.shape}")
        # logging.debug(f"cnn: {cnn_outputs}")
        # assert False
        combined_outputs = torch.cat([lstm_outputs, cnn_outputs], dim=2)
        logging.debug(f"combine shape: {combined_outputs.shape}")
        # assert False

        # CRF model
        logits = self.classifier(combined_outputs)
        # crf_outputs = self.crf.decode(logits)

        # return crf_outputs
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
        y = y.to(self.device)
        return x, y

    def loss(self, x, y_pred: torch.Tensor, y: torch.Tensor) -> float:
        mask = x['attention_mask'].bool()
        loss = -self.crf(y_pred, y, mask=mask)
        loss_item = loss.item()
        loss.backward()
        return loss_item

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, x):
        y_pred = self.forward(x)
        mask = x['attention_mask'].bool()
        logging.debug(y_pred.shape)
        y_pred = self.crf.decode(y_pred, mask=mask)
        logging.debug(y_pred)
        logging.debug(len(y_pred))
        # assert False
        return y_pred

    def train_model(self, x, y):
        y_pred = self.forward(x)
        loss_item = self.loss(x, y_pred, y)
        return loss_item
