# importing packages
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# sklearn packages
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import multilabel_confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb

# nltk packages
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download('punkt')
from string import punctuation
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from keras.preprocessing.text import Tokenizer, tokenizer_from_json

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

tags_and_type = pd.read_csv('tnt.csv')


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
Xfs = tokenizer.texts_to_matrix(tags_and_type['as_tags'], mode = 'count')
yfs = tags_and_type['type']

yfs_with_numbers = np.copy(yfs)
for i in range(len(yfs_with_numbers)):
  if (yfs_with_numbers[i] == 'present simple'):
    yfs_with_numbers[i] = 0
  if (yfs_with_numbers[i] == 'present continuous'):
    yfs_with_numbers[i] = 1
  if (yfs_with_numbers[i] == 'past simple'):
    yfs_with_numbers[i] = 2
  if (yfs_with_numbers[i] == 'future simple'):
    yfs_with_numbers[i] = 3
  if (yfs_with_numbers[i] == 'other'):
    yfs_with_numbers[i] = 4
    
train_X, test_X, train_y, test_y = train_test_split(Xfs, yfs_with_numbers, test_size=0.3, stratify=yfs_with_numbers, random_state=0)

clf = xgb.XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)

from sklearn.pipeline import Pipeline, FeatureUnion
pipe = Pipeline([('clf',clf)])

param_grid = {
     'clf__n_estimators': [300],
     
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)

grid_search.fit(train_X, train_y.astype(int))

clf_test = grid_search.best_estimator_

preds = clf_test.predict(test_X)

def print_stats(preds, target, labels, sep='-', sep_len=40, fig_size=(10,8)):
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    cm=metrics.confusion_matrix(target, preds, labels)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':fig_size})
    sns.heatmap(cm, 
        xticklabels=labels,
        yticklabels=labels,
           annot=True, cmap = 'YlGnBu')
    plt.pause(0.05)
    
print('Accuracy = %.3f' % metrics.accuracy_score(test_y.astype(int), preds.astype(int)))

print_stats(test_y.astype(int), preds.astype(int), clf_test.classes_)

#%%

input_variable = "He will have done it by this evening"
input_variable = nltk.word_tokenize(input_variable)
input_variable = [nltk.pos_tag(input_variable)]

input_variable = [[tup[1] for tup in sentence] for sentence in input_variable]

input_variable = tokenizer.texts_to_matrix(input_variable, mode = 'count')

print(clf_test.predict(input_variable))