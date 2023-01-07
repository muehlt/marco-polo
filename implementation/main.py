
from data.fetcher import Fetcher
from data.loader import Loader
from data.processor import Processor
from similarity import Similarity


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

    sim_object = Similarity(c_doc_embedded,q_doc_embedded)
    sim_data = sim_object.calc_cosine_similarity_query_docs()
    recall_dict = sim_object.recall(sim_data,0.9)
    precision_dict = sim_object.precision(sim_data, 0.9)
    f_score_dict = sim_object.f_score(recall_dict, precision_dict)

if __name__ == '__main__':
    main()