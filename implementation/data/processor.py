import nltk
from nltk.stem.snowball import SnowballStemmer
import re

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
        for idx, row in self.pd_data.iterrows():
            text = row['text']
            filtered = nltk.word_tokenize(text)
            self.pd_data['text'].at[idx] = filtered
        return self.pd_data
    
    def getStemmed(self):
        sbstemmer = SnowballStemmer("english")
        self.pd_data['text'] = self.pd_data['text'].apply(lambda txt: [sbstemmer.stem(word) for word in txt])
        return self.pd_data