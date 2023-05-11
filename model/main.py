from pipeline import Pipeline
from frontend.dataset.BIO import BIO_Dataset, BIO_DataLoader
from frontend.model.BERT_Linear import BERT_Linear
from frontend.model.BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from frontend.model.BERT_BiLSTM_CNN_CRF import BERT_BiLSTM_CNN_CRF
from frontend.model.Nezha_BiLSTM_MultiHeadSelfAttention_CNN_CRF import Nezha_BiLSTM_MultiHeadSelfAttention_CNN_CRF
from backend.optimizer.base import BaseOptimizer
from backend.analyzer.BIO import BIO_Analyzer
from utils import load_config
import logging


log_level = load_config("./config.toml", "log_level")
device = load_config("./config.toml", "device")
logging.basicConfig(format="[%(levelname)s] %(filename)s - %(funcName)s, line %(lineno)d: %(message)s", level=log_level)

# train_dataset = BIO_Dataset("/home/team_W/mmt/data/msra_ner/msra_train_bio.txt",
#                             "bert-base-chinese", "/home/team_W/mmt/framework_test/cache", split_char='\t')
# eval_dataset = BIO_Dataset("/home/team_W/mmt/data/msra_ner/msra_test_bio.txt", "bert-base-chinese",
#                            "/home/team_W/mmt/framework_test/cache", split_char='\t')
train_dataset = BIO_Dataset("/home/team_W/mmt/data/cloudfinacial/atrain.txt",
                            "bert-base-chinese", "./cache")
eval_dataset = BIO_Dataset("/home/team_W/mmt/data/cloudfinacial/adev.txt", "bert-base-chinese",
                           "./cache")

batch_size = load_config("./config.toml", "batch_size")
train_dataloader = BIO_DataLoader(train_dataset, batch_size, num_workers=8)
eval_dataloader = BIO_DataLoader(eval_dataset, batch_size, num_workers=8, drop_last=False)


model = Nezha_BiLSTM_MultiHeadSelfAttention_CNN_CRF(len(train_dataset.label2id), "sijunhe/nezha-cn-base", device,
                                                   cnn_kernel_sizes=[2, 3, 4, 5], cnn_filter_sizes=[128, 128, 128, 128])

optimizer_name, learning_rate = load_config("./config.toml", ["optimizer", "learning_rate"])
optimizer = BaseOptimizer(optimizer_name, "Exp", learning_rate, params=model.parameters(), gamma=0.99)

hyper_params = {
    "learning_rate": learning_rate,
    "batch_size"   : batch_size,
}
analyzer = BIO_Analyzer(train_dataset.id2label, hyper_params, "./cache", "sijunhe/nezha-cn-base", "hparam-test")

epochs = load_config("./config.toml", "epochs")
pipeline = Pipeline(train_dataloader, eval_dataloader, model, optimizer, analyzer, epochs, log_level)

pipeline.train()
