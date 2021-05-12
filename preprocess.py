"""""""""""""""""""""""""""""""""""""""
preprocess.py
Модуль предобработки данных.
Группа: КЭ-120
ФИО:
    Глизница Максим Николаевич 
    Снегирева Дарья Алексеевна
"""""""""""""""""""""""""""""""""""""""
import pandas as pd #используется для работы с данными
import nltk #содержит функции для обработки английского языка
import json #для сохранения объектов в формате json

from keras.preprocessing.text import Tokenizer #используется для предобработки текста

#загрузка файлов, необходимых для работы nltk
#nltk.download('punkt') 
#nltk.download('averaged_perceptron_tagger')

#%%

def sentences_to_tags(sentences, stype):
    """
    Преобразование предложений в список, состоящий из частей речи.
    
    Parameters
    ----------
    sentences : pd.DataFrame
        Набор данных, содержащий прдложения.
    stype : str
        Время предложений.

    Returns
    -------
    stc : pd.DataFrame
        Набор данных, содержащий список частей речи и время.

    """
    stc_tokd = sentences['text'].map(nltk.word_tokenize)
    stc_tagd = stc_tokd.map(nltk.pos_tag)
    stc_as_tags = [[tup[1] for tup in sentence] for sentence in stc_tagd]
    stc = pd.DataFrame()
    stc['as_tags'] = stc_as_tags
    stc['type'] = stype
    return stc

def get_convert_and_tag(data, tags, stype):
    '''
    Отбор и обработка предложений по тегу.

    Parameters
    ----------
    data : pd.DataFrame
        Набор исходных данных.
    tags : pd.DataFrame
        Набор исходных тегов.
    stype : str
        Время предложений.

    Returns
    -------
    stc : pd.DataFrame
        Набор данных, содержащий список частей речи и время.

    '''
    stc_tags = tags[tags['tag'] == stype]
    stc = data[data['id'].isin(stc_tags['id'])]
    stc = sentences_to_tags(stc, stype)
    return stc

#%%
#загрузка данных из файлов

data = pd.read_csv('eng_sentences.tsv', sep = '\t', names = ['id', 'lang', 'text'])
tags = pd.read_csv('tags.csv', sep = '\t', names = ['id', 'tag'])

#%%
#определение списка используемых времен
stypes = ['present simple', 'present continuous', 'past simple', 'present perfect', 'present perfect continuous', 'future simple']
#отбор предложений на основе списка времен
tags_and_type = pd.concat([get_convert_and_tag(data, tags, i) for i in stypes])

#%%
#замена тегов редких времен на 'other'
mask = tags_and_type.type == 'present perfect'
column_name = 'type'
tags_and_type.loc[mask, column_name] = 'other'
mask = tags_and_type.type == 'present perfect continuous'
tags_and_type.loc[mask, column_name] = 'other'

#%%
#перевод списка тегов в строку
tags_and_type['as_tags'] = tags_and_type['as_tags'].map(' '.join)

#%%
#сохранение в файл
tags_and_type.to_csv('tnt.csv', index = False)
#загрузка из файла
tags_and_type = pd.read_csv('tnt.csv')

#%%
#создание и обучение токенайзера
tokenizer = Tokenizer(num_words = 50, filters = '')
tokenizer.fit_on_texts(tags_and_type['as_tags'])

#%%
#сохранение токенайзера в файл в формате json
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))