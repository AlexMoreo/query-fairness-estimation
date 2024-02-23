import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import quapy as qp
import quapy.functional as F
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC
from quapy.protocol import AbstractProtocol
from quapy.data.base import LabelledCollection

from glob import glob
from os.path import join
from tqdm import tqdm

"""
In this third experiment, we have pairs of (Li,Ui) with Li a training set and Ui a test set as 
in the second experiment, but in this case the fairness group are defined upon geographic info.  
"""

def methods():
    yield ('CC', ClassifyAndCount(LogisticRegression()))
    yield ('PACC', PACC(LogisticRegression(), val_split=5, n_jobs=-1))
    yield ('EMQ', EMQ(LogisticRegression()))
    yield ('PCC', PCC(LogisticRegression()))
    yield ('ACC', ACC(LogisticRegression(), val_split=5, n_jobs=-1))
    yield ('MLPE', MaximumLikelihoodPrevalenceEstimation())


def load_txt_sample(path, parse_columns, verbose=False, max_lines=None):
    # print('reading', path)
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
        rank = rank[y != 'Antarctica']
        scores = scores[y != 'Antarctica']

    X = X[y!='Antarctica']
    y = y[y!='Antarctica']

    if parse_columns:
        order = np.argsort(rank)
        X = X[order]
        y = y[order]
        rank = rank[order]
        scores = scores[order]

    if max_lines is not None:
        X = X[:max_lines]
        y = y[:max_lines]

    return X, y


class RetrievedSamples(AbstractProtocol):

    def __init__(self, path_dir: str, load_fn, vectorizer, max_train_lines=None, max_test_lines=None):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.vectorizer = vectorizer
        self.max_train_lines = max_train_lines
        self.max_test_lines = max_test_lines

    def __call__(self):
        for file in glob(join(self.path_dir, 'test_rankings_*.txt')):

            X, y = self.load_fn(file.replace('test_', 'training_'), parse_columns=True, max_lines=self.max_train_lines)
            X = self.vectorizer.transform(X)
            train_sample = LabelledCollection(X, y)

            X, y = self.load_fn(file, parse_columns=True, max_lines=self.max_test_lines)
            if len(X)!=qp.environ['SAMPLE_SIZE']:
                print(f'[warning]: file {file} contains {len(X)} instances (expected: {qp.environ["SAMPLE_SIZE"]})')
            # assert len(X) == qp.environ['SAMPLE_SIZE'], f'unexpected sample size for file {file}, found {len(X)}'
            X = self.vectorizer.transform(X)
            try:
                test_sample = LabelledCollection(X, y, classes=train_sample.classes_)
            except ValueError as e:
                print(f'file {file} caused error {e}')
                yield None, None

            # print('train #classes:', train_sample.n_classes, train_sample.prevalence())
            # print('test  #classes:', test_sample.n_classes, test_sample.prevalence())

            yield train_sample, test_sample


RANK_AT_K = 100
REDUCE_TR = 50000
qp.environ['SAMPLE_SIZE'] = RANK_AT_K

data_path = './newCollectionGeo'
train_path = join(data_path, 'train_data_continent.txt')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10)

training = LabelledCollection.load(train_path, loader_func=load_txt_sample, verbose=True, parse_columns=False)

if REDUCE_TR>0:
    print('Reducing the number of documents in the training to', REDUCE_TR)
    training = training.sampling(REDUCE_TR)

Xtr, ytr = training.Xy
Xtr = tfidf.fit_transform(Xtr)
print('L orig shape = ', Xtr.shape)

training = LabelledCollection(Xtr, ytr)
classes = training.classes_

print('training classes:', classes)
print('training prevalence:', training.prevalence())

experiment_prot = RetrievedSamples(data_path,
                                   load_fn=load_txt_sample,
                                   vectorizer=tfidf,
                                   max_train_lines=None,
                                   max_test_lines=RANK_AT_K)

for method_name, quantifier in methods():
    print('Starting with method=', method_name)

    errors = []
    pbar = tqdm(experiment_prot(), total=49)
    for train, test in pbar:
        if train is not None:
            try:
                # print('Training prevalence:', F.strprev(training.prevalence()), 'shape', train.X.shape)
                # print('Test prevalence:', F.strprev(test.prevalence()), 'shape', test.X.shape)

                # print(train.prevalence())
                # print(test.prevalence())
                quantifier.fit(train)
                estim_prev = quantifier.quantify(test.instances)
                mae = qp.error.mae(test.prevalence(), estim_prev)
                errors.append(mae)
            except Exception as e:
                print(f'wow, something happened here! skipping; {e}')
        else:
            print('skipping one!')

        pbar.set_description(f'mae={np.mean(errors):.4f}')
    print()




