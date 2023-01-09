import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
import gensim
import numpy as np

class Processor:
    def __init__(self, pd_data):
        self.pd_data = pd_data
        nltk.download('stopwords')
        nltk.download('punkt')

    def getPunctuationRemoved(self):
        self.pd_data['text'] = self.pd_data['text'].apply(lambda x: re.sub(r"[^\sa-zA-Z0-9]+", '', x))
        return self.pd_data

    def getStopwordRemoved(self):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        self.pd_data['text'] = self.pd_data['text'].apply(lambda x: " ".join([word for word in x.split() if not word in stopwords]))
        return self.pd_data

    def getTokenized(self):
        self.pd_data['text'] = self.pd_data['text'].apply(lambda txt: nltk.word_tokenize(txt)) # .lower() if should be used for pretrained w2v model
        return self.pd_data

    def getStemmed(self):
        sbstemmer = SnowballStemmer("english")
        self.pd_data['text'] = self.pd_data['text'].apply(lambda txt: [sbstemmer.stem(word) for word in txt])
        return self.pd_data

    def getEmbedded(self):
        model = gensim.models.Word2Vec.load("word2vec.model")
        self.pd_data['text'] = self.pd_data['text'].apply(lambda txt: [model.wv[word] for word in txt])
        return self.pd_data

    def getDocEmbedded(self):
        # returns 1 averaged vector for each document/query
        self.pd_data['text'] = self.pd_data['text'].apply(lambda vecs: np.mean(np.array(vecs), axis=0))
        return self.pd_data

    def doPreprocessingStack(self):
        self.getPunctuationRemoved()
        self.getStopwordRemoved()
        self.getTokenized()
        self.getStemmed()
        self.getEmbedded()
        self.getDocEmbedded()
        return self.pd_data

    def getData(self):
        return self.pd_data

    def removeTextsCorpus(self):
        doc_indices = []
        doc_ids = []
        for index, row in self.pd_data.iterrows():
            if row['text'].count('.') / len(row['text']) >= 0.1:
                doc_indices.append(index)
                doc_ids.append(row['text'])

        for index in doc_indices:
            self.pd_data.drop(index, inplace=True)

        return doc_ids

    def removeTextIdsTest(self,doc_ids):
        test = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/qrels/test.tsv", sep="\t"))
        doc_indices = []
        for index, row in test.iterrows():
            for doc_id in doc_ids:
                if doc_id == row['corpus-id']:
                    doc_indices.append(index)
        for index in doc_indices:
            test.drop(index, inplace=True)

        return test

