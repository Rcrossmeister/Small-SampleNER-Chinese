# CFSC-NER
__The details of this dataset is a copy and an english translation from [here](https://github.com/Ya-dongLi/CFSC)__

## DataSet

* Data download path :

https://github.com/Ya-dongLi/CFSC/tree/main/CFSC-NER

* Data distribution :

| Financial entities types | Corporate | Stock | Market | Economy | Total |
| :----------------------: | :-------: | :---: | :----: | :-----: | :---: |
|    Number of entities    |   3841    | 11113 |  6621  |  7606   | 29181 |

* Data description(original) :

```
CFSC-NER-Alldata.json contains all data, and the json folder contains data that has been divided into training, validation, and testing sets. The original dataset is in the form of a list, with each element in the list being a JSON-formatted data. "text" represents the text of the sentence, and "labels" is a list inside the JSON, which contains several dictionaries. Each dictionary contains information about the financial entity in the sentence, where "text" is the text of the financial entity, "Level 1 Aspect" is the category of the financial entity, "from" is the starting position of the entity in the sentence, and "to" is the ending position of the entity in the sentence.
```

* Data example :

```
[
  {
    "text": "联想集团2022财年第四季度营收166.9亿美元，预估176亿美元，净利润4.12亿美元，预估3.536亿美元。",
    "labels": [
      {
        "text": "联想集团",
        "Level 1 Aspect": "Corporate",
        "from": 0,
        "to": 4
      }
    ]
  },
  {
    "text": "指数低开高走，沪指拉升涨逾1%，深成指涨1.49%，创业板指涨2.92%。",
    "labels": [
      {
        "text": "沪指",
        "Level 1 Aspect": "Stock",
        "from": 7,
        "to": 9
      },
      {
        "text": "深成指",
        "Level 1 Aspect": "Stock",
        "from": 16,
        "to": 19
      },
      {
        "text": "创业板指",
        "Level 1 Aspect": "Stock",
        "from": 26,
        "to": 30
      }
    ]
  }
]
```
* Label strategies : 

```
实	B-Corporate
丰	I-Corporate
文	I-Corporate
化	I-Corporate
公	O
告	O
，	O
公	O
司	O
与	O
中	B-Corporate
科	I-Corporate
翎	I-Corporate
碳	I-Corporate
签	O
订	O
了	O
《	O
战	O
略	O
合	O
作	O
·
·
·
```

* Reference :

https://github.com/Ya-dongLi/CFSC

## Result

_Coming soon_
