"""""""""""""""""""""""""""""""""""""""
rnn_ttb_load.py
Модуль графического интерфейса генератора.
Группа: КЭ-120
ФИО:
    Глизница Максим Николаевич 
    Снегирева Дарья Алексеевна
"""""""""""""""""""""""""""""""""""""""
# Отключение сообщений от tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf # Для работы с моделью генератора

# Настройка распределения видеопамяти
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Слои, присутствующие в модели генератора
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

import pickle # Для загрузки сериализованных файлов

import nltk # Для работы с английским языком
import pattern.en as pt # Для работы с английским языком
import random # Для получения случайных элементов списка


#%% Загрузка сериализованных файлов

vocab = pickle.load(open('vocab.pickle', 'rb'))
char_id = pickle.load(open('char_id.pickle', 'rb'))
id_char = pickle.load(open('id_char.pickle', 'rb'))

#%% Загрузка модели

vocab_size = len(vocab)
embed_dim = 128

model = Sequential()
model.add(Embedding(vocab_size, embed_dim, batch_input_shape = [1, None]))
model.add(GRU(512, return_sequences = True, stateful = True, recurrent_initializer = 'glorot_uniform'))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences = True, stateful = True, recurrent_initializer = 'glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))

model.load_weights('./models/ps-512x0.2x256x0.2-04-1.1029.hdf5')

model.build(tf.TensorShape([1, None]))

#%% Функции, выполняемые с помощью модели

def generate(model, start, num_to_gen = 100):
    """
    Генерация произвольного количества символов, начиная с определённой строки.

    Parameters
    ----------
    model : tensorflow.python.keras.engine.sequential.Sequential
        Модель генератора.
    start : str
        Начальная строка.
    num_to_gen : int, optional
        Количество символов, которое будет сгенерировано.

    """
    input_eval = [char_id[s] for s in start]
    input_eval = tf.expand_dims(input_eval, 0)
    
    gen_text = []
    temperature = 0.5
    
    model.reset_states()
    print(start, end = '')
    for i in range(num_to_gen):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        print(id_char[predicted_id], end = '')
        if(id_char[predicted_id] in ['.', '?', '!']):
            print('\n', end = '')
        #gen_text.append(id_char[predicted_id])
    
    #return start + ''.join(gen_text)

def generate_sentence(temperature):
    """
    Генерация предложения.

    Parameters
    ----------
    temperature: float
        Температура генерации.
        
    """
    input_eval = [char_id[s] for s in '. ']
    input_eval = tf.expand_dims(input_eval, 0)
    
    gen_text = []
    
    model.reset_states()
    for i in range(100):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)

        gen_text.append(id_char[predicted_id])
        if(id_char[predicted_id] in ['.', '?', '!']):
            break
    
    return ''.join(gen_text)

def generate_exercise(temperature):
    """
    Генерация упражнения.

    Parameters
    ----------
    temperature: float
        Температура генерации.
        
    """
    stc = generate_sentence(temperature)
    tgs = nltk.pos_tag(nltk.word_tokenize(stc))
    tgs = [[i[0], i[1]] for i in tgs]
    result = ''
    opt = []
    for tg in tgs:
        if tg[1] in ['VB']:
            result += ' '
            result += '<???>'
            opts = pt.lexeme(tg[0])
            if tg[0] in opts:
                opts.remove(tg[0])
            opt += [tg[0], random.sample(opts, min(2, len(opts)))]
        elif (tg[1] in ['.', '?', '!'] or "'" in tg[0]):
            result += tg[0]    
        else:
            result += ' '
            result += tg[0]
    result = result[1:]
    return result, opt

#%%