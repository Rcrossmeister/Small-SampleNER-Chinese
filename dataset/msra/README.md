# MSRA-NER
__The details of this dataset is a copy and an english translation from [here](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/MSRA)__

## Dataset

* Data download path : 

Officially : https://www.microsoft.com/en-us/download/

* Data description :

```
Language: Chinese

Tags: LOC(Location Name), ORG(Organization Name), PER(Person Name)

Tag Strategy：BIO

Split: '\t' (北\tB-LOC)
```

* Data size :

1. **[Train data set]():**  

| Sentences | Characters |  LOC  |  ORG  |  PER  |
| :-------: | :--------: | :---: | :---: | :---: |
|   45000   |  2171573   | 36860 | 20584 | 17615 |

2. **[Test data set]():**

| Sentences | Characters | LOC  | ORG  | PER  |
| :-------: | :--------: | :--: | :--: | :--: |
|   3442    |   172601   | 2886 | 1331 | 1973 |

***Where The last 3 parts represent the number of LOC, ORG, PER tags***

* Reference :

[The third international Chinese language processing bakeoff: Word segmentation and named entity recognition](https://faculty.washington.edu/levow/papers/sighan06.pdf)

https://github.com/dox1994/nlp_datasets

## Result

``` 
Epoch: 100 
processed 172601 tokens with 6190 phrases; found: 5929 phrases; correct: 5241.
accuracy:  98.24%; precision:  88.40%; recall:  84.67%; FB1:  86.49
              LOC: precision:  91.52%; recall:  88.98%; FB1:  90.23  2806
              ORG: precision:  83.04%; recall:  82.79%; FB1:  82.92  1327
              PER: precision:  87.47%; recall:  79.62%; FB1:  83.36  1796
```

