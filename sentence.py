import numpy as np
import pandas as pd
import nltk

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

#%%

'''
a = tags['tag'].unique()
a = pd.DataFrame(a)
a[a[0].str.contains('present')]
'''

data = pd.read_csv('eng_sentences.tsv', sep = '\t', names = ['id', 'lang', 'text'])

#%%

tags = pd.read_csv('tags.csv', sep = '\t', names = ['id', 'tag'])

#%%

psim_tags = tags[tags['tag'] == 'present simple']
pcon_tags = tags[tags['tag'] == 'present continuous']
pssim_tags = tags[tags['tag'] == 'past simple']
ppfc_tags = tags[tags['tag'] == 'present perfect']
ppfccon_tags = tags[tags['tag'] == 'present perfect continuous']
fsim_tags = tags[tags['tag'] == 'future simple']

#%%

psim = data[data['id'].isin(psim_tags['id'])]
pcon = data[data['id'].isin(pcon_tags['id'])]
pssim = data[data['id'].isin(pssim_tags['id'])]
ppfc = data[data['id'].isin(ppfc_tags['id'])]
ppfccon = data[data['id'].isin(ppfccon_tags['id'])]
fsim = data[data['id'].isin(fsim_tags['id'])]

#%%

ex_tokd = nltk.word_tokenize(psim.iloc[0]['text'])
ex_tagd = nltk.pos_tag(ex_tokd)