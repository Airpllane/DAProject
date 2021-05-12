import pandas as pd
import numpy as np
import nltk
import xgboost as xgb
import json
from keras.preprocessing.text import tokenizer_from_json

clf = xgb.XGBClassifier()
clf.load_model('XGB.json')

with open('tokenizer.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))

#%%

input_variable = pd.read_csv('eng_sentences.tsv', sep = '\t', names = ['id', 'lang', 'text']).iloc[:10]


#%%

#input_variable = pd.DataFrame([[1, "I walk my dog"], [2, "I am going there"]], columns = ['id', 'text'])
input_variable['tok'] = input_variable['text'].map(nltk.word_tokenize)
input_variable['tok'] = input_variable['tok'].map(nltk.pos_tag)

input_variable['tok'] = [[tup[1] for tup in sentence] for sentence in input_variable['tok']]

input_variable['tok'] = pd.Series(list(tokenizer.texts_to_matrix(input_variable['tok'], mode = 'count')))
#print(input_variable['text'].map((lambda x: tokenizer.texts_to_matrix(x, mode = 'count')))[0])

input_variable['prediction'] = clf.predict(np.array(input_variable['tok'].values.tolist()))

del input_variable['tok']

input_variable.to_csv('preds.csv', index = False)