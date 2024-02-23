import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

import quapy as qp
import quapy.functional as F
from Retrieval.commons import RetrievedSamples, load_txt_sample
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.protocol import AbstractProtocol
from quapy.data.base import LabelledCollection

from glob import glob
from os.path import join
from tqdm import tqdm

"""
In this fourth experiment, we have pairs of (Li,Ui) with Li a training set and Ui a test set as 
in the third experiment, and the fairness group are defined upon geographic info as in the third case.
The difference here is that the data Li and Ui have been drawn by retrieving query-related documents from
a pool of the same size.

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
    yield ('PACC', PACC(cls(classifier_trained), val_split=5, n_jobs=-1))
    yield ('EMQ', EMQ(cls(classifier_trained), exact_train_prev=True))
    yield ('EMQh', EMQ(cls(classifier_trained), exact_train_prev=False))
    yield ('EMQ-BCTS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='bcts'))
    yield ('EMQ-TS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='ts'))
    yield ('EMQ-NBVS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='nbvs'))
    # yield ('EMQ-VS', EMQ(cls(classifier_trained), exact_train_prev=False, recalib='vs'))
    yield ('PCC', PCC(cls(classifier_trained)))
    yield ('ACC', ACC(cls(classifier_trained), val_split=5, n_jobs=-1))
    yield ('KDE001', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.001))
    yield ('KDE005', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.005)) # <-- wow!
    yield ('KDE01', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.01))
    yield ('KDE02', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.02))
    yield ('KDE03', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.03))
    yield ('KDE05', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.05))
    yield ('KDE07', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.07))
    yield ('KDE10', KDEyML(cls(classifier_trained), val_split=5, n_jobs=-1, bandwidth=0.10))
    yield ('MLPE', MaximumLikelihoodPrevalenceEstimation())


def train_classifier():
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10)
    training = LabelledCollection.load(train_path, loader_func=load_txt_sample, verbose=True, parse_columns=False)

    if REDUCE_TR > 0:
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



RANK_AT_K = 1000
REDUCE_TR = 50000
qp.environ['SAMPLE_SIZE'] = RANK_AT_K

data_path = './50_50_split_trec'
train_path = join(data_path, 'train_50_50_continent.txt')

tfidf, classifier_trained = qp.util.pickled_resource('classifier.pkl', train_classifier)
trained=True

experiment_prot = RetrievedSamples(data_path,
                           load_fn=load_txt_sample,
                           vectorizer=tfidf,
                           max_train_lines=None,
                           max_test_lines=RANK_AT_K, classes=classifier_trained.classes_)

result_mae_dict = {}
result_mrae_dict = {}
for method_name, quantifier in methods(classifier_trained):
    # print('Starting with method=', method_name)

    mae_errors = []
    mrae_errors = []
    pbar = tqdm(experiment_prot(), total=49)
    for train, test in pbar:
        if train is not None:
            try:

                # print(train.prevalence())
                # print(test.prevalence())
                if trained and method_name!='MLPE':
                    quantifier.fit(train, val_split=train, fit_classifier=False)
                else:
                    quantifier.fit(train)
                estim_prev = quantifier.quantify(test.instances)

                mae = qp.error.mae(test.prevalence(), estim_prev)
                mae_errors.append(mae)

                mrae = qp.error.mrae(test.prevalence(), estim_prev)
                mrae_errors.append(mrae)

                # print()
                # print('Training prevalence:', F.strprev(train.prevalence()), 'shape', train.X.shape)
                # print('Test prevalence:', F.strprev(test.prevalence()), 'shape', test.X.shape)
                # print('Estim prevalence:', F.strprev(estim_prev))

            except Exception as e:
                print(f'wow, something happened here! skipping; {e}')
        else:
            print('skipping one!')

        pbar.set_description(f'{method_name}\tmae={np.mean(mae_errors):.4f}\tmrae={np.mean(mrae_errors):.4f}')
    print()
    result_mae_dict[method_name] = np.mean(mae_errors)
    result_mrae_dict[method_name] = np.mean(mrae_errors)

print('Results\n'+('-'*100))
for method_name in result_mae_dict.keys():
    MAE = result_mae_dict[method_name]
    MRAE = result_mrae_dict[method_name]
    print(f'{method_name}\t{MAE=:.5f}\t{MRAE=:.5f}')







