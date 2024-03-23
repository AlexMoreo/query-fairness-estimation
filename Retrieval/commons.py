import pandas as pd
import numpy as np
from glob import glob
from os.path import join

from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol
import json


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


def load_json_sample(path, class_name, max_lines=-1):
    obj = json.load(open(path, 'rt'))
    keys = [f'{id}' for id in range(len(obj['text'].keys()))]
    text = [obj['text'][id] for id in keys]
    classes = [obj[class_name][id] for id in keys]
    if max_lines is not None and max_lines>0:
        text = text[:max_lines]
        classes = classes[:max_lines]
    return text, classes


class TextRankings:

    def __init__(self, path, class_name):
        self.obj = json.load(open(path, 'rt'))
        self.class_name = class_name

    def get_sample_Xy(self, sample_id, max_lines=-1):
        sample_id = str(sample_id)
        O = self.obj
        docs_ids = [doc_id for doc_id, query_id in O['qid'].items() if query_id == sample_id]
        texts = [O['text'][doc_id] for doc_id in docs_ids]
        labels = [O['continent'][doc_id] for doc_id in docs_ids]
        if max_lines > 0 and len(texts) > max_lines:
            ranks = [int(O['rank'][doc_id]) for doc_id in docs_ids]
            sel = np.argsort(ranks)[:max_lines]
            texts = np.asarray(texts)[sel]
            labels = np.asarray(labels)[sel]

        return texts, labels


def get_query_id_from_path(path, prefix='training', posfix='200SPLIT'):
    qid = path
    qid = qid[:qid.index(posfix)]
    qid = qid[qid.index(prefix)+len(prefix):]
    return qid


class RetrievedSamples(AbstractProtocol):

    def __init__(self, path_dir: str, load_fn, vectorizer, max_train_lines=None, max_test_lines=None, classes=None, class_name=None):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.vectorizer = vectorizer
        self.max_train_lines = max_train_lines
        self.max_test_lines = max_test_lines
        self.classes=classes
        assert class_name is not None, 'class name should be specified'
        self.class_name = class_name
        self.text_samples = TextRankings(join(self.path_dir, 'testRankingsRetrieval.json'), class_name=class_name)


    def __call__(self):

        for file in glob(join(self.path_dir, 'training*SPLIT.json')):

            X, y = self.load_fn(file, class_name=self.class_name, max_lines=self.max_train_lines)
            X = self.vectorizer.transform(X)
            train_sample = LabelledCollection(X, y, classes=self.classes)

            query_id = get_query_id_from_path(file)
            X, y = self.text_samples.get_sample_Xy(query_id, max_lines=self.max_test_lines)

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