import pandas as pd
import numpy as np
from glob import glob
from os.path import join

import quapy.functional as F


Ks = [50, 100, 500, 1000]

CLASS_NAMES = ['continent', 'gender', 'years_category'] # ['relative_pageviews_category', 'num_sitelinks_category']:

DATA_SIZES = ['10K', '50K', '100K', '500K', '1M', 'FULL']

protected_group = {
    'gender': 'Female',
    'continent': 'Africa',
    'years_category': 'Pre-1900s',
}


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


def binarize_labels(labels, positive_class=None):
    if positive_class is not None:
        protected_labels = labels==positive_class
        labels[protected_labels] = 1
        labels[~protected_labels] = 0
        labels = labels.astype(int)
    return labels


class RetrievedSamples:
    def __init__(self,
                 class_home: str,
                 test_rankings_path: str,
                 test_query_prevs_path: str,
                 vectorizer,
                 class_name,
                 positive_class=None,
                 classes=None,
                 ):
        self.class_home = class_home
        self.test_rankings_df = pd.read_json(test_rankings_path)
        self.test_query_prevs_df = pd.read_json(test_query_prevs_path)
        self.vectorizer = vectorizer
        self.class_name = class_name
        self.positive_class = positive_class
        self.classes = classes

    def get_text_label_score(self, df, filter_rank=1000):
        df = df[df['rank']<filter_rank]

        class_name = self.class_name
        vectorizer = self.vectorizer
        filter_classes = self.classes

        text = df.text.values
        labels = df[class_name].values
        rel_score = df.score.values

        labels = binarize_labels(labels, self.positive_class)

        if filter_classes is not None:
            idx = np.isin(labels, filter_classes)
            text = text[idx]
            labels = labels[idx]
            rel_score = rel_score[idx]

        if vectorizer is not None:
            text = vectorizer.transform(text)

        order = np.argsort(-rel_score)
        return text[order], labels[order], rel_score[order]

    def __call__(self):
        tests_df = self.test_rankings_df
        class_name = self.class_name

        for file in self._list_queries():

            # loads the training sample
            train_df = pd.read_json(file)
            if len(train_df) == 0:
                print('empty dataframe: ', file)
            else:
                Xtr, ytr, score_tr = self.get_text_label_score(train_df)

                # loads the test sample
                query_id = self._get_query_id_from_path(file)
                sel_df = tests_df[tests_df.qid == query_id]
                Xte, yte, score_te = self.get_text_label_score(sel_df)

                # gets the prevalence of all judged relevant documents for the query
                df = self.test_query_prevs_df
                q_rel_prevs = df.loc[df.id == query_id][class_name+'_proportions'].values[0]

                if self.positive_class is not None:
                    if self.positive_class not in q_rel_prevs:
                        print(f'positive class {self.positive_class} not found in the query; skipping')
                        continue
                    q_rel_prevs = F.as_binary_prevalence(q_rel_prevs[self.positive_class])
                else:
                    q_rel_prevs = np.asarray([q_rel_prevs.get(class_i, 0.) for class_i in self.classes])

                yield (Xtr, ytr, score_tr), (Xte, yte, score_te), q_rel_prevs

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
        qid = int(qid)
        return qid


