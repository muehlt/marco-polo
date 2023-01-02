
from data.fetcher import Fetcher
from data.loader import Loader
from data.processor import Processor


def main():
    fetcher = Fetcher()
    fetcher.fullSetup()
    
    loader = Loader()
    loader.loadTuples(1000, 1000)
    corpus = loader.getCorpus()
    queries = loader.getQueries()

    corpus_processor = Processor(corpus)
    c_no_punctuations = corpus_processor.getPunctuationRemoved()
    c_no_stopwords    = corpus_processor.getStopwordRemoved()
    c_tokenized       = corpus_processor.getTokenized()
    c_stemmed         = corpus_processor.getStemmed()

    print(c_stemmed)

if __name__ == '__main__':
    main()