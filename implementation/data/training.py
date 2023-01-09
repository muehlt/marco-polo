
# I trained a word2vec model here with our sentences
# beforehand so we can use it later on for word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from processor import Processor
from loader import Loader
import pandas as pd
from multiprocessing import cpu_count

# gensim "sentence" is a list of words, can also be a document, so we don't have to split as long as
# the belong together contextually
# we train with the stemmed data to reduce dimension and we only need it as such later on

# since word2vec oftern classifies better if more data is provided, we introduced the
# option to extend the training data with the data that is not used in the processing
# the extended data also includes data tuples used in the evaluation processing
NR_EXTENDED_TUPLES = 0 # e.g. 100000

print("Loading data...")
loader = Loader(use_reduced=True)
loader.loadTuples()
corpus = loader.getCorpus()
queries = loader.getQueries()
summaries = loader.getSummaries()

# Extend with data that is not used in the processing later to extend word2vec performance
ext_loader = Loader(use_reduced=False)
ext_loader.loadTuples(corpus_slice=slice(NR_EXTENDED_TUPLES), query_slice=slice(NR_EXTENDED_TUPLES))
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

model = Word2Vec(sentences=context_words, vector_size=300, window=30, min_count=1, workers=cpu_count(), sg=1)
model.save("word2vec.model")
