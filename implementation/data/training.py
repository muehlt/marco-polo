
# I trained a word2vec model here with our sentences
# beforehand so we can use it later on for word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from processor import Processor
from loader import Loader
import pandas as pd

# gensim "sentence" is a list of words, can also be a document, so we don't have to split as long as
# the belong together contextually
# we train with the stemmed data to reduce dimension and we only need it as such later on

# TODO: REPLACE WITH SPINNERS

print("Loading data...")
loader = Loader(use_reduced=True)
loader.loadTuples()
corpus = loader.getCorpus()
queries = loader.getQueries()
summaries = loader.getSummaries()
del corpus['title']
data = pd.concat([corpus, queries, summaries], axis=0, ignore_index=True)

processor = Processor(data)
print("Removing punctuation...")
no_punctuations = processor.getPunctuationRemoved()
print("Removing stopwords...")
no_stopwords    = processor.getStopwordRemoved()
print("Tokenizing...")
tokenized       = processor.getTokenized()
print("Stemming...")
stemmed         = processor.getStemmed()

context_words = stemmed['text'].tolist()

print("Training model...")
model = Word2Vec(sentences=context_words, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
