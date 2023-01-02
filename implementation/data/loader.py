import json
import pandas as pd

class Loader:
    docpath = "../data/msmarco/msmarco/"
    corpus = []
    queries = []
    NR_CORPUS_TUPLES = 1000
    NR_QUERY_TUPLES = 1000

    def __init__(self, NR_CORPUS_TUPLES, NR_QUERY_TUPLES):
        self.f_corpus = open(self.docpath + 'corpus.jsonl')
        self.f_queries = open(self.docpath + 'queries.jsonl')
        self.NR_CORPUS_TUPLES = NR_CORPUS_TUPLES
        self.NR_QUERY_TUPLES = NR_QUERY_TUPLES

    def loadTuples(self):
        corpus_lines  = self.f_corpus.readlines()
        queries_lines = self.f_queries.readlines()

        nr_actual_corpus_tuples = 0
        for line in corpus_lines[:self.NR_CORPUS_TUPLES]:
            self.corpus.append(json.loads(line))
            nr_actual_corpus_tuples += 1
        print(f"Loaded {nr_actual_corpus_tuples} corpus tuples")

        nr_actual_query_tuples = 0
        for line in queries_lines[:self.NR_QUERY_TUPLES]:
            self.queries.append(json.loads(line))
            nr_actual_query_tuples += 1
        print(f"Loaded {nr_actual_query_tuples} query tuples")
    
    def getCorpus(self):
        return pd.DataFrame(self.corpus, columns=['_id', 'title', 'text', 'metadata'])

    def getQueries(self):
        return pd.DataFrame(self.queries, columns=['_id', 'text', 'metadata'])
