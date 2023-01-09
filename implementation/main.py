import os

from data.fetcher import Fetcher
from data.loader import Loader
from data.processor import Processor
from analyzer import Analyzer
from similarity import Similarity
from data.training import Training
import numpy as np
def print_results():
    pass

def main():

    if not os.path.exists("word2vec.model"):
        Training()

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

    query_processor = Processor(queries)
    q_doc_embedded    = query_processor.doPreprocessingStack()

    summaries_processor = Processor(summaries)
    s_doc_embedded    = summaries_processor.doPreprocessingStack()

    threshold = 0.615

    # CORPUS
    c_sim_object = Similarity(c_doc_embedded,q_doc_embedded,test)
    c_sim_data = c_sim_object.calc_cosine_similarity_query_docs("corpus")
    c_recall_dict = c_sim_object.recall(c_sim_data,threshold)
    c_precision_dict = c_sim_object.precision(c_sim_data,threshold)
    #print(c_precision_dict, c_recall_dict)
    c_fscore_dict = c_sim_object.f_score(c_recall_dict, c_precision_dict)
    print(f"Corpus Recall mean {np.mean(list(c_recall_dict.values()))}")
    print(f"Corpus Precision mean: {np.mean(list(c_precision_dict.values()))}")
    print(f"Corpus F-Score mean: {np.mean(list(c_fscore_dict.values()))}")

    print("====================================")

    # SUMMARY
    s_sim_object = Similarity(s_doc_embedded,q_doc_embedded,test)
    s_sim_data = s_sim_object.calc_cosine_similarity_query_docs("summarized")
    s_recall_dict = s_sim_object.recall(s_sim_data,threshold)
    s_precision_dict = s_sim_object.precision(s_sim_data,threshold)
    s_fscore_dict = s_sim_object.f_score(s_recall_dict, s_precision_dict)
    print(f"Summary Recall mean {np.mean(list(s_recall_dict.values()))}")
    print(f"Summary Precision mean: {np.mean(list(s_precision_dict.values()))}")
    print(f"Summary F-Score mean: {np.mean(list(s_fscore_dict.values()))}")

    # plotting
    data_analyzer = Analyzer(c_recall_dict, c_precision_dict, c_fscore_dict,
                             s_recall_dict, s_precision_dict, s_fscore_dict)
    data_analyzer.scatter_plot_recall(threshold)
    data_analyzer.scatter_plot_precision(threshold)
    data_analyzer.scatter_plot_fscore(threshold)
    data_analyzer.boxplot_recall_differences(threshold)
    data_analyzer.boxplot_precision_differences(threshold)
    data_analyzer.boxplot_fscore_differences(threshold)


if __name__ == '__main__':
    main()