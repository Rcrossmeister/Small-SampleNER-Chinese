# CLUENER2020
__The details of this dataset is a copy and an english translation from [here](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/README.md)__

## Dataset

* Data download path : https://www.cluebenchmarks.com/introduce.html

Run command below to download automatically :

```shell
python download_clue_data.py --data_dir=./dataset --tasks=cluener
```

* Data Categories :

```
Data is divided into 10 categories of tags, namely: 
Address, Book, Company, Game, Government, Movie, Name, Organization, Position, Scene.
```

* Data defination :

```
Address: Province, city, district, street, house number, road, street name, village, etc. (if they appear separately, they should also be marked). The address should be marked as completely as possible, down to the finest detail.

Book: Novels, magazines, exercise books, textbooks, teaching aids, atlases, cookbooks, and other types of books that can be found in bookstores, including e-books.

Company: Companies, groups, and banks (except for the central bank and the People's Bank of China, which are government institutions), such as New Oriental, including Xinhua Net/China Military Net, etc.

Game: Common games, but attention should be paid to games adapted from novels or TV dramas. The specific context should be analyzed to determine whether it is a game or not.

Government: Includes both central and local administrative agencies. The central administrative agencies include the State Council, its departments (including ministries, committees, the People's Bank of China, and the National Audit Office), and its directly affiliated agencies (such as customs, taxation, industry and commerce, and environmental protection bureaus), as well as the military.

Movie: Movies, including some documentaries shown in movie theaters. If a movie is adapted from a book, it should be distinguished from the book based on the context of the scene.

Name: Generally refers to personal names, including characters in novels, such as Song Jiang, Wu Song, and Guo Jing, as well as nicknames of famous people that can be associated with specific individuals.
Organization: Basketball teams, football teams, orchestras, clubs, etc., as well as gangs in novels, such as Shaolin Temple, Beggar's Sect, Iron Palm Gang, Wudang, Emei, etc.

Position: Historical titles such as Governor, Prefect, and Royal Tutor, as well as modern titles such as General Manager, Journalist, CEO, Artist, and Collector.

Scene: Common tourist attractions, such as Changsha Park, Shenzhen Zoo, Ocean Park, Botanical Garden, Yellow River, Yangtze River, etc.
```

* Data distribution :

```
According to the statistics of different label categories, the data distribution of the training set is as follows (Note: all entities in a piece of data are marked; if two address entities appear in a piece of data, two data will be counted when the address category data is counted) :
```

``` 
Training set label data distribution is as follows:
```

| Sentence | Address | Book | Company | Game | Government |
| :------: | :-----: | :--: | :-----: | :--: | :--------: |
|  10748   |  2829   | 1131 |  2897   | 2325 |    1797    |

| Movie | Name | Organization | Position | Scene |
| :---: | :--: | :----------: | :------: | :---: |
| 1109  | 3661 |     3075     |   3052   | 1462  |

```
Validation set label data distribution is as follows:
```

| Sentence | Address | Book | Company | Game | Government |
| :------: | :-----: | :--: | :-----: | :--: | :--------: |
|   1343   |   364   | 152  |   366   | 287  |    244     |

| Movie | Name | Organization | Position | Scene |
| :---: | :--: | :----------: | :------: | :---: |
|  150  | 451  |     344      |   425    |  199  |

* Reference : [CLUENER2020: Fine-grained Name Entity Recognition for Chinese](https://arxiv.org/abs/2001.04351)

* Data Field Explanation :

```
Taking train.json as an example, the data is divided into two columns: text & label. The text column represents the text, while the label column represents all entities that appear in the text belonging to the 10 categories.
For example:

text: "北京勘察设计协会副会长兼秘书长周荫如"
label: {"organization": {"北京勘察设计协会": [[0, 7]]}, "name": {"周荫如": [[15, 17]]}, "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}}

In this example, organization, name, and position represent entity categories. "organization": {"北京勘察设计协会": [[0, 7]]} indicates that "北京勘察设计协会" in the original text is an entity belonging to the "organization" category, and its start_index is 0 and end_index is 7 (Note: indexing starts at 0). Similarly, "name": {"周荫如": [[15, 17]]} indicates that "周荫如" in the original text is an entity belonging to the "name" category, and its start_index is 15 and end_index is 17. "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]} indicates that "副会长" and "秘书长" in the original text are entities belonging to the "position" category, and their start_indexes are 8 and 12 respectively, and their end_indexes are 10 and 14 respectively.
```

* Data Source :

```
This data is based on the THUCTC, an open-source text classification dataset from Tsinghua University. A subset of the data was selected for fine-grained named entity annotation. The original data was sourced from Sina News RSS.
```

* Reference :

[CLUENER2020: Fine-grained Name Entity Recognition for Chinese](https://arxiv.org/abs/2001.04351)

## Result

|     实体     | bilstm+crf |
| :----------: | :--------: |
| Person Name  |   75.43    |
| Organization |   74.76    |
|   Position   |   75.63    |
|   Company    |   73.02    |
|   Address    |   51.95    |
|     Game     |   81.14    |
|  Government  |   73.45    |
|    Scene     |   57.07    |
|     Book     |   72.35    |
|    Movie     |   74.23    |
| **Overall**  |   71.35    |

More details can be found [here]()
