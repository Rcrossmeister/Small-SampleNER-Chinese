import torch
from torch.nn import LayerNorm
import torch.nn as nn
from crf import CRF
from transformers import BertModel
from typing import Optional, Tuple, List
import logging
from torchcrf import CRF as torchCRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NERModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, num_classes: int, bert_model_name: str,
                 hidden_dim: Optional[int] = 256, num_layers: Optional[int] = 2):
        """
        构造函数

        Args:
            num_classes (int): tag的数量

            bert_model_name (str): BERT模型的名字

            hidden_dim (Optional[int]): LSTM隐藏单元数量

            num_layers (Optional[int]): LSTM隐藏层层数
        """
        super().__init__()
        # self.device = device
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.1)
        self.hidden2label = nn.Linear(hidden_dim * 2, num_classes)
        self.crf = torchCRF(num_classes, batch_first=True)

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

    def switch_mode(self, mode):
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        else:
            raise ValueError(f"Invalid mode {mode}")

    # def to(self, x, y):
    #     x["input_ids"] = x["input_ids"].to(self.device)
    #     x["attention_mask"] = x["attention_mask"].to(self.device)
    #     return x, y.to(self.device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def train_model(self, x, y):
        y_pred = self.forward(x)
        loss_item = self.loss(x, y_pred, y)
        return loss_item
    
    def forward_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     y_true: torch.Tensor) -> Tuple[List[List[int]], torch.Tensor]:
        """
        计算forward和loss

        Args:
            input_ids (:class:`torch.Tensor`): 句子token化后的数据，形状应该是[batch_size, seq_length]

            attention_mask (:class:`torch.Tensor`): 表示每个句子应该关注的部分，句子部分为1，填充部分为0

            y_true (:class:`torch.Tensor`): 正确的数据
        
        Returns:
            :class:`List[List[int]]`: CRF解码后的部分

            :class:`torch.Tensor`: loss
        """
        y_pred = self.forward({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        mask = attention_mask.bool()
        y_pred = self.crf.decode(y_pred, mask=mask)
        loss = -self.crf(y_pred, y_true, mask=mask)
        return y_pred, loss



