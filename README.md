# Marco Polo

TODO: Add link to presentation here

run `main.py` using `/implementation` as base directory for relative imports

## Conclusions

- Using CBOW: Worse retrieval of summarized documents (reduced Recall and Precision)
- Using SkipGram: Precision of summarized document retrieval similar to retrieval of non-summarized corpus, Recall still worse
- Use SkipGram for small datasets, it performs better for our data in general
- Word2Vec works equally well for the reduced dataset as well as an extended dataset, so a small number of words is sufficient for the used Word2Vec model in this context.

## Roles

### Fabian:
- Analysis and Plotting
    - Result processing
    - Graphical representation
- Other
    - Idea and dataset finding
 

### Martin:
- Document summarizing
    - Model training
    - Model execution
- Other
    - Idea and dataset finding

### Matthias:
- Evaluation Pipeline
    - Metric selection
    - Metric implementation
- Other
    - Idea and dataset finding


### Thomas:

- Data Processing
    - Dataset downloader (to avoid git issues)
    - Dataset reducing, by usable data (see test.tsv)
    - Dataset cleaning
    - Data loading, processing and vectorization
- Word2Vec
    - Word2Vec model training
    - Word2Vec usage and document averaging
- Other
    - Repo maintenance
    - Idea and dataset finding
