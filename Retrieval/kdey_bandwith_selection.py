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
from quapy.protocol import UPP
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as Naive
from quapy.model_selection import GridSearchQ
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.data.base import LabelledCollection

from os.path import join
from tqdm import tqdm

from result_table.src.table import Table

"""
 
"""

data_home = 'data'

datasets = ['continent', 'gender', 'years_category'] #, 'relative_pageviews_category', 'num_sitelinks_category']

for class_name in datasets:

    train_data_path = join(data_home, class_name, 'FULL', 'classifier_training.json')  # <-------- fixed classifier
    texts, labels = load_sample(train_data_path, class_name=class_name)

    classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}.pkl')
    tfidf, classifier_trained = pickle.load(open(classifier_path, 'rb'))
    classifier_hyper = classifier_trained.get_params()
    print(f'{classifier_hyper=}')

    X = tfidf.transform(texts)
    print(f'Xtr shape={X.shape}')

    pool = LabelledCollection(X, labels)
    train, val = pool.split_stratified(train_prop=0.5, random_state=0)
    q = KDEyML(LogisticRegression())
    classifier_hyper = {'classifier__C':[classifier_hyper['C'], 0.00000001], 'classifier__class_weight':[classifier_hyper['class_weight']]}
    quantifier_hyper = {'bandwidth': np.linspace(0.01, 0.2, 20)}
    hyper = {**classifier_hyper, **quantifier_hyper}
    qp.environ['SAMPLE_SIZE'] = 100
    modsel = GridSearchQ(
        model=q,
        param_grid=hyper,
        protocol=UPP(val, sample_size=100),
        n_jobs=-1,
        error='mrae',
        verbose=True
    )
    modsel.fit(train)

    print(class_name)
    print(f'{modsel.best_params_}')
    print(f'{modsel.best_score_}')











