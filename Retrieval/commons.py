import pandas as pd
import numpy as np
from glob import glob
from os.path import join

from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol


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

    def __init__(self, path_dir: str, load_fn, vectorizer, max_train_lines=None, max_test_lines=None, classes=None):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.vectorizer = vectorizer
        self.max_train_lines = max_train_lines
        self.max_test_lines = max_test_lines
        self.classes=classes

    def __call__(self):
        for file in glob(join(self.path_dir, 'test_rankings', 'test_rankingstraining_rankings_*.txt')):

            X, y = self.load_fn(file.replace('test_', 'training_'), parse_columns=True, max_lines=self.max_train_lines)
            X = self.vectorizer.transform(X)
            train_sample = LabelledCollection(X, y, classes=self.classes)

            X, y = self.load_fn(file, parse_columns=True, max_lines=self.max_test_lines)
            # if len(X)!=qp.environ['SAMPLE_SIZE']:
            #     print(f'[warning]: file {file} contains {len(X)} instances (expected: {qp.environ["SAMPLE_SIZE"]})')
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