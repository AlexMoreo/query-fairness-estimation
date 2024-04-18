import pandas as pd
import numpy as np
from glob import glob
from os.path import join

from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol
import json


def load_sample(path, class_name, max_lines=-1):
    """
    Loads a sample json as a dataframe and returns text and labels for
    the given class_name

    :param path: path to a json file
    :param class_name: string representing the target class
    :param max_lines: if provided and > 0 then returns only the
        first requested number of instances
    :return: texts and labels for class_name
    """
    df = pd.read_json(path)
    text = df.text.values
    try:
        labels = df[class_name].values
    except KeyError as e:
        print(f'error in {path}; key {class_name} not found')
        raise e
    if max_lines is not None and max_lines>0:
        text = text[:max_lines]
        labels = labels[:max_lines]
    return text, labels


class TextRankings:

    def __init__(self, path, class_name):
        self.obj = json.load(open(path, 'rt'))
        self.class_name = class_name

    def get_sample_Xy(self, sample_id, max_lines=-1):
        sample_id = str(sample_id)
        O = self.obj
        docs_ids = [doc_id for doc_id, query_id in O['qid'].items() if query_id == sample_id]
        texts = [O['text'][doc_id] for doc_id in docs_ids]
        labels = [O[self.class_name][doc_id] for doc_id in docs_ids]
        if max_lines > 0 and len(texts) > max_lines:
            ranks = [int(O['rank'][doc_id]) for doc_id in docs_ids]
            sel = np.argsort(ranks)[:max_lines]
            texts = np.asarray(texts)[sel]
            labels = np.asarray(labels)[sel]

        return texts, labels


def filter_by_classes(X, y, classes):
    idx = np.isin(y, classes)
    return X[idx], y[idx]


class RetrievedSamples(AbstractProtocol):

    def __init__(self,
                 class_home: str,
                 test_rankings_path: str,
                 load_fn,
                 vectorizer,
                 class_name,
                 max_train_lines=None,
                 max_test_lines=None,
                 classes=None
                 ):
        self.class_home = class_home
        self.test_rankings_df = pd.read_json(test_rankings_path)
        self.load_fn = load_fn
        self.vectorizer = vectorizer
        self.class_name = class_name
        self.max_train_lines = max_train_lines
        self.max_test_lines = max_test_lines
        self.classes=classes


    def __call__(self):

        for file in self._list_queries():

            texts, y = self.load_fn(file, class_name=self.class_name, max_lines=self.max_train_lines)
            texts, y = filter_by_classes(texts, y, self.classes)
            X = self.vectorizer.transform(texts)
            train_sample = LabelledCollection(X, y, classes=self.classes)

            query_id = self._get_query_id_from_path(file)
            texts, y = self._get_test_sample(query_id, max_lines=self.max_test_lines)
            texts, y = filter_by_classes(texts, y, self.classes)
            X = self.vectorizer.transform(texts)

            try:
                test_sample = LabelledCollection(X, y, classes=train_sample.classes_)
                yield train_sample, test_sample
            except ValueError as e:
                print(f'file {file} caused an exception: {e}')
                yield None, None


    def _list_queries(self):
        return sorted(glob(join(self.class_home, 'training_Query*200SPLIT.json')))

    def _get_test_sample(self, query_id, max_lines=-1):
        df = self.test_rankings_df
        sel_df = df[df.qid==int(query_id)]
        texts = sel_df.text.values
        try:
            labels = sel_df[self.class_name].values
        except KeyError as e:
            print(f'error: key {self.class_name} not found in test rankings')
            raise e
        if max_lines > 0 and len(texts) > max_lines:
            ranks = sel_df.rank.values
            idx = np.argsort(ranks)[:max_lines]
            texts = np.asarray(texts)[idx]
            labels = np.asarray(labels)[idx]
        return texts, labels

    def total(self):
        return len(self._list_queries())

    def _get_query_id_from_path(self, path):
        prefix = 'training_Query-'
        posfix = 'Sample-200SPLIT'
        qid = path
        qid = qid[:qid.index(posfix)]
        qid = qid[qid.index(prefix) + len(prefix):]
        return qid