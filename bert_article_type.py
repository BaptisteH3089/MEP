#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:57:39 2021

@author: baptistehessel

"""
import torch
import transformers as ppb
import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import corpus_module

path_customer = '/Users/baptistehessel/Documents/DAJ/MEP/montageIA/data/CM2/'

with open(path_customer + 'dict_pages', 'rb') as f:
    dict_pages = pickle.load(f)

# load model, tokenizer and weights
camembert, tokenizer, weights = (ppb.CamembertModel,
                                 ppb.CamembertTokenizer,
                                 'camembert-base')

# Load pretrained model/tokenizer
tokenizer = tokenizer.from_pretrained(weights)
model = camembert.from_pretrained(weights)

corpus, labels = corpus_module.CreationCorpusLabels(dict_pages)

corpus, labels = corpus[:500], labels[:500]

"""
# see if there are length > 512
max_len = 0
indes = []
for i, sent in enumerate(corpus):
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    if len(input_ids) > 512:
        indes.append(i)
        print("annoying review at", i, "with length", len(input_ids))
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
"""

small_corpus = [txt[:512] for txt in corpus]

tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True),
                     small_corpus))

print("len(tokenized)", len(tokenized))

max_len = 0
for i in tokenized:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
print("np.array(padded).shape", np.array(padded).shape)

attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask.shape, "attention_mask.shape")

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()


ss = ShuffleSplit(n_splits=4)
Y = np.array(labels)
lr = LogisticRegression()

for train, test in ss.split(features, Y):
    lr.fit(features[train], Y[train])
    preds = lr.predict(features[test])
    score = f1_score(Y[test], preds, average='macro')
    print(f"score: {score:.2f}")


