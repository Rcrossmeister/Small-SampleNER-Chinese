# Small-SampleNER-Chinese
A BiLSTM-CRF model is used to complete Chinese NER tasks. It is an improvement of the baseline model from [CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/bilstm_crf_pytorch), we use it to adapt our task in the future.

**NOTIFICATION:**

_Note : This repo is built for code support of our paper, the more available methods is developing in process._

_Note : A Bert-BiLSTM-CRF version is planned to upload later soon_.

_Note : the transfer document for BIO to json is available soon._

_Note : A baseline result of CCKS2017, 2018, 2019, Weibo, etc. is planned to upload later soon._

*Note : An data interface that can modify the below dataset to adapt to this code will be uploaded.*

## DataSet

1. __[CLUENER2020](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/cluener)__

CLUENER2020 is a well-defined fine-grained dataset for named entity recognition in Chinese. It contains **10 categories** and is more challenging than current other Chinese NER datasets and could better reflect real-world scenarios. The categories include person, organization, location, product, time, event, animal, plant, food and instrument.

2. [__MSRA-NER__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/msra)

MSRA-NER dataset is a Chinese named entity recognition dataset that contains **three types of named entities**: person (PER), organization (ORG), and location (LOC). It is one of the datasets provided for the named entity recognition task in the fifth international Chinese language processing bakeoff. The dataset is in the BIO scheme.

3. [__CFSC-NER__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/cfsc)

CFSC-NER dataset provides 10667 Chinese financial news sentences with a total of 29181 financial entities, with a single sentence as one data point. The training set has 8533 samples, the validation set has 1067 samples, and the test set has 1067 samples (8:1:1). Each data point consists of sentence text, financial entity position, financial entity, and financial entity category. The maximum length of a sentence is x. **There are four financial entity categories** include Corporate, Stock, Market, and Economy. In addition to all data points, we also provide data points that are divided into training sets, validation sets, and test sets as well as BIO annotated data .

## Use

* To **train** the BiLSTM-CRF baseline model with **CLUENER2020** dataset :

```shell
python run_BiLSTM-CRF.py \
		--do_train \
		--data_dir ./dataset/cluener \
		--epochs 100 \
		--batch_size 64 \
		--embedding_size 128 \ 
```

 ___If you want to train your own data, strictly follow the format of CLUENER2020 dataset above and change the `--data_dir` to adapt yours ,more quick transfer data interface will be available soon!___

* Using the trained model to **predict**

```shell
python run_BiLSTM-CRF.py \
		--do_predict \ 
		--out_dir ./output \ 
```

More parameters can be found [here](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/blob/main/run_BiLSTM-CRF.py) for further optimization.

## Result

The following baseline result is evaluated by F1-score

|  Model\Dataset  | CLUENER2020 | MSRA  | CFSC-NER |
| :-------------: | :---------: | :---: | :------: |
|   BiLSTM-CRF    |    71.35    | 86.49 |          |
| Bert-BiLSTM-CRF |             |       |          |

## Citation

Please cite our paper if you use it in your work:

```
@inproceedings{,
   title={{}: },
   author={Zijin_Hong},
   booktitle={},
   year={2023}
}
```

