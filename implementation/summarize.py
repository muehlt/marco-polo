import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data.loader import Loader
import pandas as pd

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)

# TODO: IMPROVE PERFORMANCE
device = torch.device('cpu')

print("Loading data...")
loader = Loader()
loader.loadTuples(100, 100) # TODO: FIT TO DATA WE REALLY USE OR WHOLE CORPUS IF RESOURCES!
corpus = loader.getCorpus()

summarized_texts = []
for index, text in enumerate(corpus['text']):
    preprocessed = text.strip().replace("\n","")
    text_to_summarize = "summarize: " + preprocessed
    text_as_tensors = tokenizer.encode(text_to_summarize,return_tensors='pt').to(device)

    summary_tokens = model.generate(text_as_tensors,max_length=(len(preprocessed.split())-1))
    summary_text = tokenizer.decode(summary_tokens[0],skip_special_tokens=True)

    summarized_texts.append(summary_text)

    if(index%10 == 0):
        print(index)


corpus['summarized'] = summarized_texts

print(corpus[['text', 'summarized']])

