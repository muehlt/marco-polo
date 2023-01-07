
from data.fetcher import Fetcher
from data.loader import Loader
from data.processor import Processor
from similarity import Similarity
import matplotlib.pyplot as plt

def main():
    fetcher = Fetcher()
    fetcher.fullSetup()
    
    loader = Loader(use_reduced=True)
    loader.loadTuples()
    corpus = loader.getCorpus()
    queries = loader.getQueries()
    summaries = loader.getSummaries()

    corpus_processor = Processor(corpus)
    removed_ids = corpus_processor.removeTextsCorpus()
    test = corpus_processor.removeTextIdsTest(removed_ids)
    c_doc_embedded    = corpus_processor.doPreprocessingStack()

    #print(c_doc_embedded)

    query_processor = Processor(queries)
    q_doc_embedded    = query_processor.doPreprocessingStack()

    #print(q_doc_embedded)

    summaries_processor = Processor(summaries)
    s_doc_embedded    = summaries_processor.doPreprocessingStack()

    #print(q_doc_embedded)

    print(c_doc_embedded, s_doc_embedded)

    threshold = 0.875
    c_sim_object = Similarity(c_doc_embedded,q_doc_embedded,test)
    c_sim_data = c_sim_object.calc_cosine_similarity_query_docs()
    c_recall_dict = c_sim_object.recall(c_sim_data,threshold)
    c_precision_dict = c_sim_object.precision(c_sim_data,threshold)
    print(c_precision_dict, c_recall_dict)
    c_fscore_dict = c_sim_object.f_score(c_recall_dict, c_precision_dict)


    s_sim_object = Similarity(s_doc_embedded,q_doc_embedded,test)
    s_sim_data = s_sim_object.calc_cosine_similarity_query_docs()
    s_recall_dict = s_sim_object.recall(s_sim_data,threshold)
    s_precision_dict = s_sim_object.precision(s_sim_data,threshold)
    s_fscore_dict = s_sim_object.f_score(s_recall_dict, s_precision_dict)

    print("Corpus F-Scores: ", c_fscore_dict)
    print("Summaries F-Scores: ", s_fscore_dict)


    keys = list(range(0, 43))
    # scatter plot recall per query for corpus & summarized
    plt.figure(1)
    recall = plt.axes()
    recall.scatter(keys, c_recall_dict.values(), label="Corpus")
    recall.scatter(keys, s_recall_dict.values(), label="Summaries")
    recall.set_title(f"Recall \n t = {threshold}")
    recall.set_ylabel('Recall score')
    recall.set_xlabel('Query IDs')
    recall.legend()
    plt.show()

    # scatter plot precision per query for corpus & summarized
    plt.figure(2)
    prec = plt.axes()
    prec.scatter(keys, c_precision_dict.values(), label="Corpus")
    prec.scatter(keys, s_precision_dict.values(), label="Summaries")
    prec.set_title(f"Precision \n t = {threshold}")
    prec.set_ylabel('Precision score')
    prec.set_xlabel('Query IDs')
    prec.legend()
    plt.show()

    # scatter plot fscore per query for corpus & summarized
    plt.figure(3)
    fscore = plt.axes()
    fscore.scatter(keys, c_fscore_dict.values(), label="Corpus")
    fscore.scatter(keys, s_fscore_dict.values(), label="Summaries")
    fscore.set_title(f"F-Score \n t = {threshold}")
    fscore.set_ylabel('F-Score')
    fscore.set_xlabel('Query IDs')
    fscore.legend()
    plt.show()


if __name__ == '__main__':
    main()