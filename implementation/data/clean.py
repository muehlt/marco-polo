# Removes misplaced unicode hex from reduced corpus and html encodings

# NOTE: Although read in the right encoding (ASCII), the corpus contains non-ASCII characters
#       probably because of web scraping. These characters are misinterpreted while reading and
#       unfortunately not all these characters are replaced by the preprocessing stack 
#       (punctuation removal) because of this misinterpretation and numbers would remain.
#       Because translating the characters would be complex due to the mixed encodings with 
#       little benefit we decided to remove them and replace them entirely, which is non-disturbing 
#       in almost all cases. Also there is 1 html &nbsp (and potential other html encoded things) so
#       we resolved them using the html library.

from loader import Loader
from halo import Halo
import html

spinner = Halo(text='Cleaning data', spinner='dots')
spinner.start()

loader = Loader(use_reduced=True)
loader.loadTuples()
corpus = loader.getCorpus()

corpus["text"] = corpus.text.apply(lambda x: html.unescape(x))
corpus["text"] =  corpus.text.str.replace('[^\x00-\x7F]',' ')   # remove non-ascii characters and replace them with whitespace 
                                                                # because they will be removed by the preprocessing stack anyway
                                                                # translating would be complex due to the mixed encodings

with open ("../data/msmarco/msmarco/corpus.reduced.jsonl", 'w') as f:
    f.write(corpus.to_json(lines=True, orient="records"))

spinner.succeed("Cleaned data")