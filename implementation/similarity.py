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

    def calc_cosine_similarity_query_docs(self):

        similarity_data = pd.DataFrame()

        for query in self.queries.iterrows():
            query_results = dict()

            for doc in self.corpus.iterrows():
                cos_sim = calc_cosine_similarity(query[1]['text'], doc[1]['text'])
                query_results[doc[1]['_id']] = cos_sim

            df_query_results = pd.DataFrame([query_results])
            similarity_data = pd.concat([similarity_data, df_query_results], ignore_index=True)

        return similarity_data
