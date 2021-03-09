import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

#%%

data = pd.read_csv('tnt.csv')

print(data['type'].value_counts())

#%%

tdct = {'present simple': 1}#, 'past simple': 2, 'present continuous': 3}
data['type'] = data['type'].map(tdct).fillna(0)

#%%

X = data['as_tags']
v = CountVectorizer()

X = v.fit_transform(X).toarray()
y = data['type']

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y)

#%%

depth = 4
min_split = 50
min_leaf = 50

clf = DecisionTreeClassifier(max_depth = depth, min_samples_split = min_split, min_samples_leaf = min_leaf)

#%%

clf.fit(X_train, y_train)
print("Train accuracy: {:.2f}".format(clf.score(X_train, y_train)))
print("Test accuracy: {:.2f}".format(clf.score(X_test, y_test)))

#%%

plt.figure(figsize = (40, 20))
plot_tree(clf, filled = True, feature_names = v.get_feature_names())
plt.savefig('tree.png')
plt.show()