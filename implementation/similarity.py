import pandas as pd
import numpy as np


def calc_cosine_similarity(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))


class Similarity:

    def __init__(self, corpus, queries):
        self.corpus = corpus
        self.queries = queries

    def recall(self, data, threshold):
        result_dict = {}
        for i,col in enumerate(data.columns):
            retrieved_list = data[i].loc[data[i] >= threshold].index.tolist()
            query_id = self.queries['_id'][i]
            test = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/qrels/test.tsv", sep="\t"))
            base_relevant = test['corpus-id'].loc[test['query-id'] == query_id].tolist()
            intersection_relevant_retrieved = set(retrieved_list) & set(base_relevant)
            recall = len(intersection_relevant_retrieved) / len(base_relevant)
            result_dict[query_id] = recall
        return result_dict

    def precision(self, data, threshold):
        result_dict = {}
        for i,col in enumerate(data.columns):
            retrieved_list = data[i].loc[data[i] >= threshold].index.tolist()
            query_id = self.queries['_id'][i]
            test = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/qrels/test.tsv", sep="\t"))
            base_relevant = test['corpus-id'].loc[test['query-id'] == query_id].tolist()
            intersection_relevant_retrieved = set(retrieved_list) & set(base_relevant)
            recall = len(intersection_relevant_retrieved) / len(retrieved_list)
            result_dict[query_id] = recall
        return result_dict

    def f_score(self,recall_dict,precision_dict):
        f_score_dict = {}
        for query in enumerate(recall_dict.keys()):
            f_score = 2*precision_dict[query]*recall_dict[query] / (precision_dict[query]+recall_dict[query])
            f_score_dict[query] = f_score
        return f_score_dict

    def calc_cosine_similarity_query_docs(self):

        similarity_data = pd.DataFrame()

        for query in self.queries.iterrows():
            query_results = dict()

            for doc in self.corpus.iterrows():
                cos_sim = calc_cosine_similarity(query[1]['text'], doc[1]['text'])
                query_results[doc[1]['_id']] = cos_sim

            df_query_results = pd.DataFrame([query_results])
            similarity_data = pd.concat([similarity_data, df_query_results], ignore_index=True)
        similarity_data_transpose = similarity_data.transpose()

        return similarity_data_transpose
