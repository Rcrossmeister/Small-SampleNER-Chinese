import torch
from torch.utils.data import DataLoader
from .base import BaseDataset
from typing import Sequence, Mapping, List, Optional, Tuple, Literal
from transformers import BertTokenizer
import logging
import pickle
import os


class BIO_Dataset(BaseDataset):
    def __init__(self, file_name: str, tokenizer_name: str, cache_dir: str,
                 split_char: Optional[str] = ' ', max_seq_length: Optional[int] = 256, device: Optional[str] = "cuda:0"):
        self.data = self.load_data(file_name, split_char)
        self.max_seq_length = max_seq_length
        # self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.label2id, self.id2label = self.save_labels(cache_dir)
        self.device = device

    def load_data(self, file_name: str, split_char: Optional[str] = ' ') -> Sequence[Mapping[str, List[str]]]:
        raw = []
        with open(file_name, "r", encoding="utf-8") as file:
            sentence = []
            label = []
            for line in file:
                if line == '\n':
                    raw.append({
                        "sentence": sentence,
                        "label": label
                    })
                    sentence = []
                    label = []
                else:
                    data = line.strip().split(split_char)
                    if len(data) == 2:
                        sentence.append(data[0])
                        label.append(data[1])
        logging.info(f"len(raw): {len(raw)}")
        return raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, List[str]]:
        return self.data[index]

    def save_labels(self, cache_dir: str) -> Tuple[Mapping[str, int], Mapping[int, str]]:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        label2id = None
        id2label = None

        labels = set()
        for data in self.data:
            for i in data["label"]:
                labels.add(i)
        labels = list(labels)
        label2id = {label: id for id, label in enumerate(labels)}
        id2label = {id: label for id, label in enumerate(labels)}
        if cache_dir[-1] != '/':
            cache_dir = cache_dir + '/'

        if os.path.exists(cache_dir + "label2id.pkl"):
            logging.info("label2id.pkl exists")
            with open(cache_dir + "label2id.pkl", "rb") as file:
                label2id = pickle.load(file)
            logging.info("load success")
        else:
            with open(cache_dir + "label2id.pkl", "wb") as file:
                pickle.dump(label2id, file)
                logging.info("label2id.pkl dump success")

        if os.path.exists(cache_dir + "id2label.pkl"):
            logging.info("id2label.pkl exists")
            with open(cache_dir + "id2label.pkl", "rb") as file:
                id2label = pickle.load(file)
            logging.info("load success")
        else:
            with open(cache_dir + "id2label.pkl", "wb") as file:
                pickle.dump(id2label, file)
                logging.info("id2label.pkl dump success")

        logging.info(f"len(label): {len(label2id)}")
        return label2id, id2label

    def collate_fn(self, batch: Sequence[Mapping[str, List[str]]]) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        sentences = []
        labels = []
        logging.debug(batch)
        for i in batch:
            logging.debug(i)
            assert len(i["sentence"]) == len(i["label"])
            sentence = []
            label = []
            for j in range(len(i["sentence"])):
                token = self.tokenizer.tokenize(i["sentence"][j])
                if len(token) == 1:
                    sentence.append(token[0])
                    label.append(i["label"][j])
            sentences.append(sentence)
            labels.append(label)

        maxi = 0
        for i in sentences:
            maxi = max(len(i), maxi)
        maxi = min(self.max_seq_length - 2, maxi)

        logging.debug(f"maxi = {maxi}, max_seq_length = {self.max_seq_length}")

        for i in range(len(sentences)):
            assert len(sentences[i]) == len(labels[i])
            if len(sentences[i]) > maxi:
                del sentences[i][maxi:]
                del labels[i][maxi:]

        logging.debug(sentences)
        logging.debug(labels)
        # logging.info(sentences[-1])
        # logging.info(labels[-1])

        ids = []
        masks = []
        y = []
        maxi += 2

        for s, l in zip(sentences, labels):
            id = self.tokenizer.convert_tokens_to_ids(s)
            id = [self.tokenizer.cls_token_id] + id + [self.tokenizer.sep_token_id]
            while len(id) < maxi:
                id.append(self.tokenizer.pad_token_id)
            mask = [1 if i != self.tokenizer.pad_token_id else 0 for i in id]
            ids.append(id)
            masks.append(mask)

            label = [self.label2id[i] for i in l]
            label = [self.label2id["O"]] + label + [self.label2id["O"]]
            while len(label) < maxi:
                label.append(self.label2id["O"])
            y.append(label)

        logging.debug(ids)
        logging.debug(masks)
        logging.debug(y)

        ids = torch.tensor(ids)
        masks = torch.tensor(masks)
        y = torch.tensor(y)

        logging.debug(ids)
        logging.debug(masks)
        logging.debug(y)

        x = {
            "input_ids": ids,
            "attention_mask": masks
        }

        logging.debug(x)

        return x, y


class BIO_DataLoader(DataLoader):
    def __init__(self, dataset: BaseDataset, batch_size: int,
                 shuffle: Optional[bool] = True, num_workers: Optional[int] = 4, drop_last: Optional[bool] = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=dataset.collate_fn, drop_last=drop_last)


if __name__ == '__main__':
    # logger = logging.getLogger("DEBUG")
    logging.basicConfig(format="[%(levelname)s] %(filename)s - %(funcName)s, line %(lineno)d: %(message)s", level=logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    bio = BIO_Dataset("/home/team_W/mmt/data/weibo_NER/train.txt", "bert-base-chinese", "/home/team_W/mmt/framework/cache")
    test = bio[0:2]
    bio.collate_fn(test)


