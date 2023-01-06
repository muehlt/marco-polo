import json
import pandas as pd

class Loader:
    docpath = "../data/msmarco/msmarco/"
    corpus = []
    queries = []

    def __init__(self):
        self.f_corpus = open(self.docpath + 'corpus.jsonl')
        self.f_queries = open(self.docpath + 'queries.jsonl')

    def loadTuples(self, NR_CORPUS_TUPLES, NR_QUERY_TUPLES):
        corpus_lines  = self.f_corpus.readlines()
        queries_lines = self.f_queries.readlines()

        nr_actual_corpus_tuples = 0
        for line in corpus_lines[:NR_CORPUS_TUPLES]:
            j_str_line = json.loads(line)
            j_str_line['_id'] = int(j_str_line['_id'])
            self.corpus.append(j_str_line)
            nr_actual_corpus_tuples += 1
        #print(f"Loaded {nr_actual_corpus_tuples} corpus tuples")

        nr_actual_query_tuples = 0
        for line in queries_lines[:NR_QUERY_TUPLES]:
            j_str_line = json.loads(line)
            j_str_line['_id'] = int(j_str_line['_id'])
            self.queries.append(j_str_line)
            nr_actual_query_tuples += 1
        #print(f"Loaded {nr_actual_query_tuples} query tuples")
    
    def getCorpus(self):
        return pd.DataFrame(self.corpus, columns=['_id', 'title', 'text', 'metadata'])

    def getQueries(self):
        return pd.DataFrame(self.queries, columns=['_id', 'text', 'metadata'])
