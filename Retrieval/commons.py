import pandas as pd
import numpy as np
from glob import glob
from os.path import join

from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol
import json


def load_sample(path, class_name):
    """
    Loads a sample json as a dataframe and returns text and labels for
    the given class_name

    :param path: path to a json file
    :param class_name: string representing the target class
    :return: texts, labels for class_name
    """
    df = pd.read_json(path)
    text = df.text.values
    labels = df[class_name].values
    return text, labels


def get_text_label_score(df, class_name, vectorizer=None, filter_classes=None):
    text = df.text.values
    labels = df[class_name].values
    rel_score = df.score.values

    if filter_classes is not None:
        idx = np.isin(labels, filter_classes)
        text = text[idx]
        labels = labels[idx]
        rel_score = rel_score[idx]

    if vectorizer is not None:
        text = vectorizer.transform(text)

    order = np.argsort(-rel_score)
    return text[order], labels[order], rel_score[order]


class RetrievedSamples:

    def __init__(self,
                 class_home: str,
                 test_rankings_path: str,
                 vectorizer,
                 class_name,
                 classes=None
                 ):
        self.class_home = class_home
        self.test_rankings_df = pd.read_json(test_rankings_path)
        self.vectorizer = vectorizer
        self.class_name = class_name
        self.classes=classes


    def __call__(self):
        tests_df = self.test_rankings_df
        class_name = self.class_name
        vectorizer = self.vectorizer

        for file in self._list_queries():

            # print(file)

            # loads the training sample
            train_df = pd.read_json(file)
            if len(train_df) == 0:
                print('empty dataframe: ', file)
            else:
                Xtr, ytr, score_tr = get_text_label_score(train_df, class_name, vectorizer, filter_classes=self.classes)

                # loads the test sample
                query_id = self._get_query_id_from_path(file)
                sel_df = tests_df[tests_df.qid == int(query_id)]
                Xte, yte, score_te = get_text_label_score(sel_df, class_name, vectorizer, filter_classes=self.classes)

                yield (Xtr, ytr, score_tr), (Xte, yte, score_te)

    def _list_queries(self):
        return sorted(glob(join(self.class_home, 'training_Query*200SPLIT.json')))

    # def _get_test_sample(self, query_id, max_lines=-1):
    #     df = self.test_rankings_df
    #     sel_df = df[df.qid==int(query_id)]
    #     return get_text_label_score(sel_df)
        # texts = sel_df.text.values
        # try:
        #     labels = sel_df[self.class_name].values
        # except KeyError as e:
        #     print(f'error: key {self.class_name} not found in test rankings')
        #     raise e
        # if max_lines > 0 and len(texts) > max_lines:
        #     ranks = sel_df.rank.values
        #     idx = np.argsort(ranks)[:max_lines]
        #     texts = np.asarray(texts)[idx]
        #     labels = np.asarray(labels)[idx]
        # return texts, labels

    def total(self):
        return len(self._list_queries())

    def _get_query_id_from_path(self, path):
        prefix = 'training_Query-'
        posfix = 'Sample-200SPLIT'
        qid = path
        qid = qid[:qid.index(posfix)]
        qid = qid[qid.index(prefix) + len(prefix):]
        return qid


