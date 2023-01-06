
from data.fetcher import Fetcher
from data.loader import Loader
from data.processor import Processor


def main():
    fetcher = Fetcher()
    fetcher.fullSetup()
    
    loader = Loader(use_reduced=True)
    loader.loadTuples()
    corpus = loader.getCorpus()
    queries = loader.getQueries()

    corpus_processor = Processor(corpus)
    c_doc_embedded    = corpus_processor.doPreprocessingStack()

    print(c_doc_embedded)

    query_processor = Processor(queries)
    q_doc_embedded    = query_processor.doPreprocessingStack()

    print(q_doc_embedded)

if __name__ == '__main__':
    main()