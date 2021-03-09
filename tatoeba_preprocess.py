import numpy as np
import pandas as pd
import nltk

from keras.preprocessing.text import Tokenizer

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

#%%

def sentences_to_tags(sentences, stype):
    stc_tokd = sentences['text'].map(nltk.word_tokenize)
    stc_tagd = stc_tokd.map(nltk.pos_tag)
    stc_as_tags = [[tup[1] for tup in sentence] for sentence in stc_tagd]
    stc = pd.DataFrame()
    stc['as_tags'] = stc_as_tags
    stc['type'] = stype
    return stc

def get_convert_and_tag(data, tags, stype):
    stc_tags = tags[tags['tag'] == stype]
    stc = data[data['id'].isin(stc_tags['id'])]
    stc = sentences_to_tags(stc, stype)
    return stc

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

stypes = ['present simple', 'present continuous', 'past simple', 'present perfect', 'present perfect continuous', 'future simple']

tags_and_type = pd.concat([get_convert_and_tag(data, tags, i) for i in stypes])

tags_and_type.to_csv('tnt.csv', index = False)

#%%

tokenizer = Tokenizer(num_words = 50, filters = '')
tokenizer.fit_on_texts(tags_and_type['as_tags'])
Xfs = tokenizer.texts_to_matrix(tags_and_type['as_tags'], mode = 'count')
yfs = tags_and_type['type']