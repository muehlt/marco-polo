import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data.loader import Loader
from data.processor import Processor
from tqdm import tqdm
import pandas as pd

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)

# check if we can use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading data...")
loader = Loader(use_reduced=True)
loader.loadTuples()
corpus = loader.getCorpus()
corpus_processor = Processor(corpus)
corpus_processor.removeTextsCorpus()

progress_bar = tqdm(total=len(corpus), desc=f"Generating summaries using {device}", unit="docs")


def summarize(text):
    # remove newlines and leading spaces
    preprocessed = text.strip().replace("\n", "")
    # encode
    text_as_tensors = tokenizer.encode("summarize: " + preprocessed, return_tensors='pt').to(device)
    # run summary model
    summary_tokens = model.generate(text_as_tensors, max_length=(len(preprocessed.split()) - 1))
    # decode
    summary_text = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    progress_bar.update(1)
    return summary_text


corpus['text'] = corpus['text'].apply(summarize)

progress_bar.close()


# write summarized docs to file
with open("../data/msmarco/msmarco/summaries.jsonl", 'w') as file:
    file.write(corpus.to_json(lines=True, orient='records'))


