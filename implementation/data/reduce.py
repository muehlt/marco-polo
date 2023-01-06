# Reduce dataset to a smaller size (only corpus and query tuples we need in test.tsv)

import pandas as pd

from loader import Loader
from halo import Halo

spinner = Halo(text='Reducing data', spinner='dots')
spinner.start()

test = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/qrels/test.tsv", sep="\t")) 
loader = Loader()
loader.loadTuples()
corpus = loader.getCorpus()
queries = loader.getQueries()

# if using tsv for better performance
# corpus = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/corpus.tsv", sep="\t"))
# queries = pd.DataFrame(pd.read_csv("../data/msmarco/msmarco/queries.tsv", sep="\t"))

# automatically removes tuples without score
test = test.merge(corpus, left_on='corpus-id', right_on='_id')
test = test.merge(queries, left_on='query-id', right_on='_id')

corpus_reduced = test[['_id_x', 'title', 'text_x', 'metadata_x']]
corpus_reduced.rename(columns={'_id_x': '_id', 'text_x': 'text', 'metadata_x': 'metadata'}, inplace=True)
queries_reduced = test[['_id_y', 'text_y', 'metadata_y']]
queries_reduced.rename(columns={'_id_y': '_id', 'text_y': 'text', 'metadata_y': 'metadata'}, inplace=True)

corpus_reduced.drop_duplicates(subset='_id', inplace=True)
queries_reduced.drop_duplicates(subset='_id', inplace=True)

# if output should be tsv file for better performance (not original data structure, so we use jsonl)
# corpus_reduced.to_csv("../data/msmarco/msmarco/corpus_reduced.tsv", sep="\t", index=False)
# queries_reduced.to_csv("../data/msmarco/msmarco/queries_reduced.tsv", sep="\t", index=False)

with open ("../data/msmarco/msmarco/corpus.reduced.jsonl", 'w') as f:
    f.write(corpus_reduced.to_json(lines=True, orient="records"))

with open ("../data/msmarco/msmarco/queries.reduced.jsonl", 'w') as f:
    f.write(queries_reduced.to_json(lines=True, orient="records"))

spinner.succeed("Reduced data")