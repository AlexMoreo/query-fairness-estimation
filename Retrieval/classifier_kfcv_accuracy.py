import itertools
import os.path
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

import quapy as qp
from Retrieval.commons import RetrievedSamples, load_sample
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as Naive
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.data.base import LabelledCollection

from os.path import join
from tqdm import tqdm

from result_table.src.table import Table

"""
 
"""

data_home = 'data'

datasets = ['continent', 'gender', 'years_category', 'relative_pageviews_category', 'num_sitelinks_category']

param_grid = {'C': np.logspace(-4, 4, 9), 'class_weight': ['balanced', None]}

classifiers = [
    ('LR', LogisticRegression(max_iter=5000), param_grid),
    ('SVM', LinearSVC(), param_grid)
]

def benchmark_name(class_name):
    return class_name.replace('_', '\_')

table = Table(name=f'accuracy', benchmarks=[benchmark_name(d) for d in datasets])
table.format.show_std = False
table.format.stat_test = None
table.format.lower_is_better = False
table.format.color = False
table.format.remove_zero = True
table.format.style = 'rules'

for class_name, (cls_name, cls, grid) in itertools.product(datasets, classifiers):

    train_data_path = join(data_home, class_name, 'FULL', 'classifier_training.json')  # <-------- fixed classifier

    texts, labels = load_sample(train_data_path, class_name=class_name)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3)
    Xtr = tfidf.fit_transform(texts)
    print(f'Xtr shape={Xtr.shape}')

    print('training classifier...', end='')
    classifier = GridSearchCV(
        cls,
        param_grid=grid,
        n_jobs=-1,
        cv=5,
        verbose=10
    )
    classifier.fit(Xtr, labels)
    classifier_acc = classifier.best_score_
    classifier_acc_per_fold = classifier.cv_results_['mean_test_score'][classifier.best_index_]

    print(f'[done] best-params={classifier.best_params_} got {classifier_acc:.4f} score, per fold {classifier_acc_per_fold}')

    table.add(benchmark=benchmark_name(class_name), method=cls_name, v=classifier_acc_per_fold)

    Table.LatexPDF(f'./latex/classifier_Acc.pdf', tables=[table])








