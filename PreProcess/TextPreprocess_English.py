import re
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
import nltk

class TextPreprocess_English:
    def __init__(self,text):
        self.text=text
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self):
        # 去除数字
        text = re.sub(r'\d+', '', self.text)
        # 转换为小写
        text = text.lower()
        # 去除多余空格
        # text = ' '.join(text.split())
        # 去除HTML标签
        text = BeautifulSoup(text, 'html.parser').get_text()
        # 去除特殊字符
        text = re.sub(r'[^\w\s,.]', '', text)
        return text

    def clean_symbol(self):
        self.text = re.sub(r"(?<!\w')([a-zA-Z',.]+)(?!\w)", r"\1 ", self.text)
        self.text = re.sub(r"(?<!\w')([^a-zA-Z\s',.]+)(?!\w')", "", self.text)
        self.text = re.sub(r"(?<=\w)'(?=\w)", "", self.text)
        words = self.text.split()
        self.text = " ".join(words)
        return self.text

        return self.text

    def TextPreprocess(self):
        self.text=self.clean_text()
        self.text=self.clean_symbol()
        return self.text


if __name__ == "__main__":
    f = open('/home/hzj/NLP1/SentimentAnalysis/Code/CTC/IMDB/data/tmp.txt', encoding='utf-8')
    data = f.readlines()
    f.close()
    text = ""
    for line in data:
        text += line
    print(text)
    tp = TextPreprocess_English(text)
    text = tp.TextPreprocess()
    print("result")
    print(text)