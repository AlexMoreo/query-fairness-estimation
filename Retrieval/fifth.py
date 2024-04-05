from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

import quapy as qp
import quapy.functional as F
from Retrieval.commons import RetrievedSamples, load_txt_sample, load_json_sample
from Retrieval.tabular import Table
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.protocol import AbstractProtocol
from quapy.data.base import LabelledCollection

from glob import glob
from os.path import join
from tqdm import tqdm

"""
In this fifth experiment, we have pairs of (Li,Ui) with Li a training set and Ui a test set as 
in the fourth experiment, and the fairness group are defined upon geographic info as in the fourth case.
As in the fourth, the data Li and Ui have been drawn by retrieving query-related documents from
a pool of the same size. Unlike the fourth experiment, here the training queries are

Por ahora 1000 en tr y 100 en test
Parece que ahora hay muy poco shift  
"""


def cls(classifier_trained=None):
    if classifier_trained is None:
        # return LinearSVC()
        return LogisticRegression()
    else:
        return classifier_trained


def methods(classifier_trained=None):
    yield ('CC', ClassifyAndCount(cls(classifier_trained)))
    yield ('PCC', PCC(cls(classifier_trained)))
    yield ('ACC', ACC(cls(classifier_trained), val_split=5, n_jobs=-1))
    yield ('PACC', PACC(cls(classifier_trained), val_split=5, n_jobs=-1))
    yield ('EMQ', EMQ(cls(classifier_trained), exact_train_prev=True))
    yield ('EMQh', EMQ(cls(classifier_trained), exact_train_prev=False))
    # yield ('EMQ-BCTS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='bcts'))
    # yield ('EMQ-TS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='ts'))
    # yield ('EMQ-NBVS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='nbvs'))
    # yield ('EMQ-VS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='vs'))
    # yield ('KDE001', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.001))
    # yield ('KDE005', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.005)) # <-- wow!
    # yield ('KDE01', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.01))
    # yield ('KDE02', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.02))
    # yield ('KDE03', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.03))
    # yield ('KDE05', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.05))
    yield ('KDE07', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.07))
    # yield ('KDE10', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.10))
    yield ('MLPE', MaximumLikelihoodPrevalenceEstimation())


def train_classifier():
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10)
    training = LabelledCollection.load(train_path, loader_func=load_json_sample, class_name=CLASS_NAME)

    if REDUCE_TR > 0 and len(training) > REDUCE_TR:
        print('Reducing the number of documents in the training to', REDUCE_TR)
        training = training.sampling(REDUCE_TR, *training.prevalence())

    Xtr, ytr = training.Xy
    Xtr = tfidf.fit_transform(Xtr)
    print('L orig shape = ', Xtr.shape)

    training = LabelledCollection(Xtr, ytr)

    print('training classifier')
    classifier_trained = LogisticRegression()
    classifier_trained = GridSearchCV(classifier_trained,
                                      param_grid={'C': np.logspace(-3, 3, 7), 'class_weight': ['balanced', None]},
                                      n_jobs=-1, cv=5)
    classifier_trained.fit(Xtr, ytr)
    classifier_trained = classifier_trained.best_estimator_
    trained = True
    print('[Done!]')

    classes = training.classes_

    print('training classes:', classes)
    print('training prevalence:', training.prevalence())

    return tfidf, classifier_trained


def reduceAtK(data: LabelledCollection, k):
    X, y = data.Xy
    X = X[:k]
    y = y[:k]
    return LabelledCollection(X, y, classes=data.classes_)


RANK_AT_K = -1
REDUCE_TR = 50000
qp.environ['SAMPLE_SIZE'] = RANK_AT_K

data_path = {
    'first_letter_category': './first_letter_categoryDataset',
    'continent': './newExperimentalSetup'
}

def scape_latex(string):
    return string.replace('_', '\_')


Ks = [10, 50, 100, 250, 500, 1000, 2000]
# Ks = [500]

for CLASS_NAME in ['first_letter_category']: #['continent']: #, 'gender', 'gender_category', 'occupations', 'source_countries', 'source_subcont_regions', 'years_category', 'relative_pageviews_category']:

    train_path = join(data_path[CLASS_NAME], 'train3000samples.json')

    tfidf, classifier_trained = qp.util.pickled_resource(f'classifier_{CLASS_NAME}.pkl', train_classifier)
    trained=True

    experiment_prot = RetrievedSamples(data_path[CLASS_NAME],
                               load_fn=load_json_sample,
                               vectorizer=tfidf,
                               max_train_lines=None,
                               max_test_lines=RANK_AT_K, classes=classifier_trained.classes_, class_name=CLASS_NAME)

    method_names = [name for name, *other in methods()]
    benchmarks = [f'{scape_latex(CLASS_NAME)}@{k}' for k in Ks]
    table_mae = Table(benchmarks, method_names, color_mode='global')
    table_mrae = Table(benchmarks, method_names, color_mode='global')

    for method_name, quantifier in methods(classifier_trained):
        # print('Starting with method=', method_name)

        mae_errors = {k:[] for k in Ks}
        mrae_errors = {k:[] for k in Ks}

        pbar = tqdm(experiment_prot(), total=49)
        for train, test in pbar:
            if train is not None:
                try:
                    if trained and method_name!='MLPE':
                        quantifier.fit(train, val_split=train, fit_classifier=False)
                    else:
                        quantifier.fit(train)

                    for k in Ks:
                        test_k = reduceAtK(test, k)
                        estim_prev = quantifier.quantify(test_k.instances)

                        mae_errors[k].append(qp.error.mae(test_k.prevalence(), estim_prev))
                        mrae_errors[k].append(qp.error.mrae(test_k.prevalence(), estim_prev, eps=(1./(2*k))))

                except Exception as e:
                    print(f'wow, something happened here! skipping; {e}')
            else:
                print('skipping one!')

            # pbar.set_description(f'{method_name}\tmae={np.mean(mae_errors):.4f}\tmrae={np.mean(mrae_errors):.4f}')
            pbar.set_description(f'{method_name}')

        for k in Ks:

            table_mae.add(benchmark=f'{scape_latex(CLASS_NAME)}@{k}', method=method_name, values=mae_errors[k])
            table_mrae.add(benchmark=f'{scape_latex(CLASS_NAME)}@{k}', method=method_name, values=mrae_errors[k])

    table_mae.latexPDF('./latex', f'table_{CLASS_NAME}_mae.tex')
    table_mrae.latexPDF('./latex', f'table_{CLASS_NAME}_mrae.tex')








