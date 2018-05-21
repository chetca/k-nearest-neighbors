from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import operator
import pylab as pl

url = r'https://archive.ics.uci.edu/ml/' \
    'machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = [u'Чашелистик длина, см',
              u'Чашелистик ширина, см',
              u'Лепесток длина, см',
              u'Лепесток ширина, см',
             'Class']
#print(df)

def test_and_train(df, proportion):
    mask = np.random.rand(len(df)) < proportion
    return df[mask], df[~mask]
train, test = test_and_train(df, 0.67)

from math import sqrt
def euclidean_distance(instance1,instance2):
    squares = [(i-j)**2 for i,j in zip(instance1,instance2)]
    return sqrt(sum(squares))

def get_neighbours(instance, train,k):
    distances = []
    for i in train.ix[:,:-1].values:
        distances.append(euclidean_distance(instance,i))
    distances = tuple(zip(distances, train[u'Class'].values))
    return sorted(distances,key=operator.itemgetter(0))[:k]

from collections import Counter
def get_response(neigbours):
    return Counter(neigbours).most_common()[0][0][1]

def get_predictions(train, test, k):
    predictions = []
    for i in test.ix[:,:-1].values:
        neigbours = get_neighbours(i,train,k)
        response = get_response(neigbours)
        predictions.append(response)
    return predictions

def mean(instance):
    return sum(instance)/len(instance)
def get_accuracy(test,predictions):
    return mean([i == j for i,j in zip(test[u'Class'].values, predictions)])
get_accuracy(test,get_predictions(train, test, 5))

#Юзаем KNeighborsClassifier

variables = [u'Чашелистик длина, см',u'Чашелистик ширина, см',
              u'Лепесток длина, см',u'Лепесток ширина, см']
results = []
for n in range(1,51,1):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[variables], train[u'Class'])
    preds = clf.predict(test[variables])
    accuracy = np.where(preds==test[u'Class'], 1, 0).sum() / float(len(test))
    print("Neighbors: %d, Accuracy: %3f" % (n, accuracy))
    results.append([n, accuracy])
results = pd.DataFrame(results, columns=["n", "accuracy"])

sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'Class', size = 6)\
.map(plt.scatter, u'Лепесток длина, см', u'Лепесток ширина, см')\
.add_legend()
plt.show()
