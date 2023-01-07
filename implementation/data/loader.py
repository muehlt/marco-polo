import json
import pandas as pd

class Loader:
    docpath = "../data/msmarco/msmarco/"
    corpus = []
    queries = []
    summaries = []

    def __init__(self, use_reduced=False):
        self.f_corpus = open(self.docpath + ('corpus.reduced.jsonl' if use_reduced else 'corpus.jsonl'))
        self.f_queries = open(self.docpath + ('queries.reduced.jsonl' if use_reduced else 'queries.jsonl'))
        self.f_summaries = open(self.docpath + ('summaries.jsonl'))

    def loadTuples(self, corpus_slice=slice(None, None, 1), query_slice=slice(None, None, 1)):
        corpus_lines  = self.f_corpus.readlines()
        queries_lines = self.f_queries.readlines()

        nr_actual_corpus_tuples = 0
        for line in corpus_lines[corpus_slice]:
            j_str_line = json.loads(line)
            j_str_line['_id'] = int(j_str_line['_id'])
            self.corpus.append(j_str_line)
            nr_actual_corpus_tuples += 1
        #print(f"Loaded {nr_actual_corpus_tuples} corpus tuples")

        nr_actual_query_tuples = 0
        for line in queries_lines[query_slice]:
            j_str_line = json.loads(line)
            j_str_line['_id'] = int(j_str_line['_id'])
            self.queries.append(j_str_line)
            nr_actual_query_tuples += 1
        #print(f"Loaded {nr_actual_query_tuples} query tuples")

        # load summaries
        for line in self.f_summaries.readlines():
            j_str_line = json.loads(line)
            j_str_line['_id'] = int(j_str_line['_id'])
            self.summaries.append(j_str_line)
    
    def getCorpus(self):
        return pd.DataFrame(self.corpus, columns=['_id', 'title', 'text', 'metadata'])

    def getQueries(self):
        return pd.DataFrame(self.queries, columns=['_id', 'text', 'metadata'])

    def getSummaries(self):
        return pd.DataFrame(self.summaries, columns=['_id', 'text', 'metadata'])
