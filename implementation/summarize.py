import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data.loader import Loader
from tqdm import tqdm
import pandas as pd

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)

# check if we can use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading data...")
loader = Loader()
loader.loadTuples(100, 100) # TODO: FIT TO DATA WE REALLY USE OR WHOLE CORPUS IF RESOURCES!
corpus = loader.getCorpus()

progress_bar = tqdm(total=len(corpus), desc=f"Generating summaries using {device}", unit="docs")
summarized_texts = []
for text in corpus['text']:
    preprocessed = text.strip().replace("\n","")
    text_to_summarize = "summarize: " + preprocessed
    text_as_tensors = tokenizer.encode(text_to_summarize,return_tensors='pt').to(device)

    summary_tokens = model.generate(text_as_tensors,max_length=(len(preprocessed.split())-1))
    summary_text = tokenizer.decode(summary_tokens[0],skip_special_tokens=True)

    summarized_texts.append(summary_text)

    progress_bar.update(1)

progress_bar.close()


corpus['summarized'] = summarized_texts

print(corpus[['text', 'summarized']])

