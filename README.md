# Small-SampleNER-Chinese
A BiLSTM-CRF model is used to complete Chinese NER tasks. It is an improvement of the baseline model from [CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/bilstm_crf_pytorch), we use it to adapt our task in the future.

**NOTIFICATION:**

_Note : This repo is built for code support of our paper, the more available methods is developing in process._

_Note : A baseline result of CCKS2017, 2018, 2019, Weibo, etc. is planned to upload later soon._

*Note : An data interface for CCKS is still empty, will be available soon* 

*Note : The detail version of dataset description and README will be updated soon* 

__Update:__

* _The transfer document for BIO to json is available now. 05-13-2023_
* *The raw version of Bert-BiLSTM-CRF is available, will fix it later*

## DataSet

1. __[CLUENER2020](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/cluener)__

CLUENER2020 is a well-defined fine-grained dataset for named entity recognition in Chinese. It contains **10 categories** and is more challenging than current other Chinese NER datasets and could better reflect real-world scenarios. The categories include person, organization, location, product, time, event, animal, plant, food and instrument.

2. [__MSRA-NER__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/msra)

MSRA-NER dataset is a Chinese named entity recognition dataset that contains **three types of named entities**: person (PER), organization (ORG), and location (LOC). It is one of the datasets provided for the named entity recognition task in the fifth international Chinese language processing bakeoff. The dataset is in the BIO scheme.

3. [__CFSC-NER__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/cfsc)

CFSC-NER dataset provides 10667 Chinese financial news sentences with a total of 29181 financial entities, with a single sentence as one data point. The training set has 8533 samples, the validation set has 1067 samples, and the test set has 1067 samples (8:1:1). Each data point consists of sentence text, financial entity position, financial entity, and financial entity category. The maximum length of a sentence is x. **There are four financial entity categories** include Corporate, Stock, Market, and Economy. In addition to all data points, we also provide data points that are divided into training sets, validation sets, and test sets as well as BIO annotated data .

4. [__CCKS2017__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/ccks2017)

CCKS2017 is a dataset for Chinese clinical named entity recognition (NER), which is a fundamental and critical task for other natural language processing (NLP) tasks. The dataset consists of 1198 records for training and 398 records for testing

**--need interface**

4. [__CCKS2018__](https://github.com/Rcrossmeister/Small-SampleNER-Chinese/tree/main/dataset/ccks2018)

CCKS2018 is a named entity recognition (NER) task focusing on Chinese electronic medical records (EMR). The dataset provided by CCKS2018 medical NER academic competition consists of 600 entries of real-world CEMRs as training dataset

**--need interface**

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

|  Model\Dataset    | CLUENER2020 | MSRA  | CFSC-NER | CCKS2017 | CCKS2018 |
| :-------------:   | :---------: | :---: | :------: | :------: | :------: |
|   BiLSTM-CRF      |    71.35    | 88.29 |   72.4   |          |          |
|synomous(m=3,r=0.3)|             | 93.37 |   71.4   |          |          |
|delete  (m=3,r=0.3)|             | 89.83 |          |          |          |
|  Bert-BiLSTM-CRF  |    78.38    | 94.51 |   74.44  |   91.46  |   81.96  |
|     Bert-CRF      |    79.03    | 95.59 |   74.41  |   91.82  |   82.29  |

active_learn (each_query raito = 0.2) on BiLSTM-CRF

|  Dataset\Ratio    | 0.2(initial)| 0.4   | 0.6      | 0.8      | 1.0      |
| :-------------:   | :---------: | :---: | :------: | :------: | :------: |
|   CLUENER2020(r)  |    57.61    | 64.04 |  68.34   |   70.13  |  71.41   |
|   CLUENER2020(lc) |    57.71    | 65.11 |  68.74   |   71.05  |  70.81   |
|       MSRA(r)     |    79.53    | 84.21 |  86.64   |   87.29  |  87.76   |
|       MSRA(lc)    |    78.43    | 84.56 |  86.84   |   87.13  |  87.91   |
|     CFSC-NER(r)   |    66.26    | 69.84 |  71.16   |   72.54  |  72.63   |
|    CFSC-NER(lc)   |    66.21    | 69.27 |  71.21   |   71.39  |  72.71   |
|     CCKS2017      |    00.00    |       |          |          |          |
|     CCKS2017      |    00.00    |       |          |          |          |

## Citation

Please cite our paper if you use it in your work:

```
@inproceedings{,
   title={{}: },
   author={},
   booktitle={},
   year={2023}
}
```

