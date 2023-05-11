import datetime
import logging
from abc import abstractmethod
from .base import BaseAnalyzer
from typing import Mapping, Literal, Any, Dict
from comet_ml import Experiment
from visualdl import LogWriter


class BIO_Analyzer(BaseAnalyzer):
    def __init__(self, id2label: Mapping[int, str], hyper_params: Dict, cache_path: str,
                 model_name: str, mode: Literal["normal", "hparam-test"]):
        self.loss = None
        self.total_loss = 0
        self.epoch_loss = 0
        self.epoch_step = 0
        self.epoch_num = 1
        self.step = 0
        self.f1 = None
        self.best_f1 = None
        self.best_overall_f1 = None
        self.y_pred = []
        self.y_true = []
        self.id2label = id2label
        self.exp = Experiment(
            api_key="sMHPGYLjlokfE6PSp1eq5YUKp",
            project_name="self-atten-test",
            workspace="maidangdang1",
        )
        self.exp.log_parameters(hyper_params)

        timestamp = str(datetime.datetime.now()).split('.')[0]
        if cache_path[-1] != '/':
            cache_path += '/'
        self.logger = LogWriter(cache_path + '/'.join([model_name, mode, timestamp]))
        self.logger.add_hparams(hyper_params, ["eval/f1"])


    def update_loss(self, loss):
        self.loss = loss
        self.total_loss += loss
        self.epoch_loss += loss
        self.step += 1
        self.epoch_step += 1
        self.exp.log_metric("train/loss", loss, step=self.step)
        self.logger.add_scalar("train/loss", loss, step=self.step)

    def calc_eval_value(self):
        help_set = set([None])
        label = sorted(list(set([i[2:] if len(i) > 2 else None for i in self.id2label.values()]) - help_set))
        label.append("OVERALL")
        gold_label = {i: 0 for i in label}
        predict_label = {i: 0 for i in label}
        correct_label = {i: 0 for i in label}
        p = {i: 0 for i in label}
        r = {i: 0 for i in label}
        f1 = {i: 0 for i in label}
        logging.debug(self.id2label)

        for i in range(len(self.y_pred)):
            for j in range(len(self.y_pred[i])):
                logging.debug(self.y_pred[i][j])
                self.y_pred[i][j] = [self.id2label[k] for k in self.y_pred[i][j]]

        for i in range(len(self.y_true)):
            for j in range(len(self.y_true[i])):
                self.y_true[i][j] = [self.id2label[k] for k in self.y_true[i][j]]

        for i, j in zip(self.y_pred, self.y_true):
            for k, l in zip(i, j):
                y_pred_set = set()
                y_true_set = set()
                flag = False
                start = 0
                # end = 0
                tag = ""

                # assert len(k) == len(l), f"len(k) = {len(k)}, len(l) = {len(l)}"
                # logging.debug(f"len(k) = {len(k)}, len(l) = {len(l)}")

                for m in range(min(len(k), len(l))):
                    # logging.debug(k[m])
                    if k[m] == "O":
                        if flag:
                            y_pred_set.add((start, m, tag))
                            flag = False
                    elif "B-" in k[m]:
                        if flag:
                            y_pred_set.add((start, m, tag))
                        start = m
                        tag = k[m][2:]
                        flag = True
                    elif "I-" in k[m]:
                        if flag:
                            if k[m][2:] != tag:
                                flag = False
                    else:
                        raise ValueError()
                if flag:
                    y_pred_set.add((start, len(k), tag))

                start = 0
                tag = ""
                flag = False
                for m in range(min(len(k), len(l))):
                    # logging.debug(l[m])
                    if l[m] == "O":
                        if flag:
                            y_true_set.add((start, m, tag))
                            # logging.debug((start, m, tag))
                            flag = False
                    elif "B-" in l[m]:
                        if flag:
                            y_true_set.add((start, m, tag))
                        start = m
                        tag = l[m][2:]
                        flag = True
                    elif "I-" in l[m]:
                        if flag:
                            if l[m][2:] != tag:
                                flag = False
                    else:
                        raise ValueError()
                if flag:
                    y_true_set.add((start, len(k), tag))

                for i in y_true_set:
                    gold_label[i[2]] += 1
                for i in y_pred_set:
                    predict_label[i[2]] += 1
                # logging.debug(y_true_set)
                logging.debug(y_pred_set)
                correct_set = y_pred_set & y_true_set
                for i in correct_set:
                    correct_label[i[2]] += 1

        gold_label["OVERALL"] = sum(gold_label.values())
        predict_label["OVERALL"] = sum(predict_label.values())
        correct_label["OVERALL"] = sum(correct_label.values())
        # assert False
        logging.debug(gold_label)
        # logging.info(f"gold_label: {gold_label}")
        for i in label:
            p[i] = correct_label[i] / predict_label[i] if predict_label[i] != 0 else 0
            r[i] = correct_label[i] / gold_label[i] if gold_label[i] != 0 else 0
            f1[i] = (2 * p[i] * r[i]) / (p[i] + r[i]) if (p[i] + r[i]) != 0 else 0
        return label, p, r, f1

    def update_eval_value(self, x, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true.cpu().tolist())

    def update_epoch(self) -> None:
        avg_loss = self.total_loss / self.step if self.step > 0 else None
        epoch_avg_loss = self.epoch_loss / self.epoch_step if self.epoch_step > 0 else None
        self.exp.log_metric("train/avg_loss", avg_loss, epoch=self.epoch_num)
        self.exp.log_metric("train/epoch_avg_loss", epoch_avg_loss, epoch=self.epoch_num)
        self.exp.log_metric("eval/f1", self.f1["OVERALL"], epoch=self.epoch_num)

        self.logger.add_scalar("train/avg_loss", avg_loss, step=self.epoch_num)
        self.logger.add_scalar("train/epoch_avg_loss", epoch_avg_loss, step=self.epoch_num)
        self.logger.add_scalar("eval/f1", self.f1["OVERALL"], step=self.epoch_num)

        self.epoch_loss = 0
        self.epoch_step = 0
        self.epoch_num += 1
        self.y_pred = []
        self.y_true = []

    def get_loss(self):
        avg_loss = self.total_loss / self.step if self.step > 0 else None
        epoch_avg_loss = self.epoch_loss / self.epoch_step if self.epoch_step > 0 else None
        return self.loss, avg_loss, epoch_avg_loss

    def get_f1(self):
        return self.f1, self.best_f1

    def get_eval_value(self, value: str):
        pass

    def print_loss(self):
        _, avg_loss, epoch_avg_loss = self.get_loss()
        logging.info(f"train/avg_loss: {avg_loss:.2f}, train/epoch_avg_loss: {epoch_avg_loss:.2f}")

    def print_eval_value(self):
        label, p, r, f1 = self.calc_eval_value()
        self.f1 = f1
        if self.best_f1 is None:
            self.best_f1 = f1
        elif sum([i for i in f1.values()]) - f1["OVERALL"] > sum([j for j in self.best_f1.values()]) - self.best_f1["OVERALL"]:
            self.best_f1 = f1

        if self.best_overall_f1 is None:
            self.best_overall_f1 = f1
        elif f1["OVERALL"] > self.best_overall_f1["OVERALL"]:
            self.best_overall_f1 = f1

        logging.info(f"{'label'.ljust(16)}{'p'.ljust(16)}{'r'.ljust(16)}{'f1'.ljust(16)}{'best_f1'.ljust(16)}"
                     f"{'best_overall_f1'.ljust(16)}")
        for i in label:
            info = f"{i}".ljust(16)\
                   + f"{p[i]:.2f}".ljust(16)\
                   + f"{r[i]:.2f}".ljust(16)\
                   + f"{f1[i]:.2f}".ljust(16)\
                   + f"{self.best_f1[i]:.2f}".ljust(16)\
                   + f"{self.best_overall_f1[i]:.2f}".ljust(16)
            logging.info(info)
        # assert False













