import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import LinearSVC

from quapy.data.base import LabelledCollection
from sklearn.model_selection import cross_val_score, GridSearchCV

from os.path import join

"""
In this experiment, I simply try to understand whether the learning task can be learned or not.
The problem is that we are quantifying the categories based on the alphabetical order (of what?).  
"""

def load_txt_sample(path, parse_columns, verbose=False, max_lines=None):
    if verbose:
        print(f'loading {path}...', end='')
    df = pd.read_csv(path, sep='\t')
    if verbose:
        print('[done]')
    X = df['text'].values
    y = df['continent'].values

    if parse_columns:
        rank = df['rank'].values
        scores = df['score'].values
        order = np.argsort(rank)
        X = X[order]
        y = y[order]
        rank = rank[order]
        scores = scores[order]

    if max_lines is not None:
        X = X[:max_lines]
        y = y[:max_lines]

    return X, y

data_path = './50_50_split_trec'
train_path = join(data_path, 'train_50_50_continent.txt')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10)
data = LabelledCollection.load(train_path, loader_func=load_txt_sample, verbose=True, parse_columns=False)
data = data.sampling(20000)
train, test = data.split_stratified()
train.instances = tfidf.fit_transform(train.instances)
test.instances  = tfidf.transform(test.instances)

# svm = LinearSVC()
# cls = GridSearchCV(svm, param_grid={'C':np.logspace(-3,3,7), 'class_weight':['balanced', None]})
cls = LogisticRegression()
cls.fit(*train.Xy)

# score = cross_val_score(LogisticRegressionCV(), *data.Xy, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1, cv=5)
# print(score)
# print(np.mean(score))

y_pred = cls.predict(test.instances)
macrof1 = f1_score(y_true=test.labels, y_pred=y_pred, average='macro')
microf1 = f1_score(y_true=test.labels, y_pred=y_pred, average='micro')

print('macro', macrof1)
print('micro', microf1)
