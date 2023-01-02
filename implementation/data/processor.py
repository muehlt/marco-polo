import nltk
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
        for idx, row in self.pd_data.iterrows():
            text = row['text']
            filtered = re.sub(r"[^\sa-zA-Z0-9]+", '', text)
            self.pd_data['text'].at[idx] = filtered
        return self.pd_data

    def getStopwordRemoved(self):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        for idx, row in self.pd_data.iterrows():
            text = row['text']
            filtered = " ".join([word for word in text.split() if not word in stopwords])
            self.pd_data['text'].at[idx] = filtered
        return self.pd_data
    
    def getTokenized(self):
        self.pd_data['text'] = self.pd_data['text'].apply(lambda txt: nltk.word_tokenize(txt))
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
        self.pd_data['text'] = self.pd_data['text'].apply(lambda vecs: np.mean( np.array(vecs), axis=0 ))
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