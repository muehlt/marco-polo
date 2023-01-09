
# I trained a word2vec model here with our sentences
# beforehand so we can use it later on for word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from processor import Processor
from loader import Loader
import pandas as pd
import os

# gensim "sentence" is a list of words, can also be a document, so we don't have to split as long as
# the belong together contextually
# we train with the stemmed data to reduce dimension and we only need it as such later on

print("Loading data...")
loader = Loader(use_reduced=True)
loader.loadTuples()
corpus = loader.getCorpus()
queries = loader.getQueries()
summaries = loader.getSummaries()

# Extend with data that is not used in the processing later to extend word2vec performance
ext_loader = Loader(use_reduced=False)
ext_loader.loadTuples(corpus_slice=slice(100000), query_slice=slice(100000))
ext_corpus = loader.getCorpus()
ext_queries = loader.getQueries()

del corpus['title']
del ext_corpus['title']
data = pd.concat([corpus, queries, summaries, ext_corpus, ext_queries], axis=0, ignore_index=True)

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

model = Word2Vec(sentences=context_words, vector_size=300, window=30, min_count=1, workers=4,sg=1)
model.save("word2vec.model")
