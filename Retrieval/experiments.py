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
In this sixth experiment, we have a collection C of >6M documents.
We split C in two equally-sized pools TrPool, TePool

I have randomly split the collection in 50% train and 50% split. In each split we have approx. 3.25 million documents. 

We have 5 categories we can evaluate over: Continent, Years_Category, Num_Site_Links, Relative Pageviews and Gender. 

From the training set I have created smaller subsets for each category:
100K, 500K, 1M and FULL (3.25M) 

For each category and subset, I have created a training set called: "classifier_training.json". This is the "base" training set for the classifier. In this set we have 500 documents per group in a category. (For example: Male 500, Female 500, Unknown 500).  Let me know if you think we need more. 

To "bias" the quantifier towards a query, I have executed the queries (97) on the different training sets and retrieved the 200 most relevant documents per group. 
For example: (Male 200, Female 200, Unknown 200) 
Sometimes this is infeasible, we should probably discuss this at some point. 

 You can find the results for every query in a file named: 

"training_Query-[QID]Sample-200SPLIT.json" 

Test: 
To evaluate our approach, I have executed the queries on the test split. You can find the results for all 97 queries up till k=1000 in this file. 
 testRanking_Results.json 
  
"""


def methods(classifier, class_name):

    kde_param = {
        'continent': 0.01,
        'gender': 0.005,
        'years_category':0.03
    }

    yield ('Naive', Naive())
    yield ('NaiveQuery', Naive())
    yield ('CC', ClassifyAndCount(classifier))
    # yield ('PCC', PCC(classifier))
    # yield ('ACC', ACC(classifier, val_split=5, n_jobs=-1))
    yield ('PACC', PACC(classifier, val_split=5, n_jobs=-1))
    # yield ('EMQ', EMQ(classifier, exact_train_prev=True))
    # yield ('EMQ-Platt', EMQ(classifier, exact_train_prev=True, recalib='platt'))
    # yield ('EMQh', EMQ(classifier, exact_train_prev=False))
    # yield ('EMQ-BCTS', EMQ(classifier, exact_train_prev=True, recalib='bcts'))
    # yield ('EMQ-TS', EMQ(classifier, exact_train_prev=False, recalib='ts'))
    # yield ('EMQ-NBVS', EMQ(classifier, exact_train_prev=False, recalib='nbvs'))
    # yield ('EMQ-VS', EMQ(classifier, exact_train_prev=False, recalib='vs'))
    # yield ('KDE001', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.001))
    # yield ('KDE005', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.005)) # <-- wow!
    # yield ('KDE01', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.01))
    # yield ('KDE02', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.02))
    # yield ('KDE03', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.03))
    # yield ('KDE-silver', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth='silverman'))
    # yield ('KDE-scott', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth='scott'))
    yield ('KDEy-ML', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=kde_param[class_name]))
    # yield ('KDE005', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.005))
    yield ('KDE01', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.01))
    # yield ('KDE02', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.02))
    # yield ('KDE03', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.03))
    # yield ('KDE04', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.04))
    # yield ('KDE05', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.05))
    # yield ('KDE07', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.07))
    # yield ('KDE10', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.10))


def train_classifier(train_path):
    """
    Trains a classifier. To do so, it loads the training set, transforms it into a tfidf representation.
    The classifier is Logistic Regression, with hyperparameters C (range [0.001, 0.01, ..., 1000]) and
    class_weight (range {'balanced', None}) optimized via 5FCV.

    :return: the tfidf-vectorizer and the classifier trained
    """
    texts, labels = load_sample(train_path, class_name=class_name)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3)
    Xtr = tfidf.fit_transform(texts)
    print(f'Xtr shape={Xtr.shape}')

    print('training classifier...', end='')
    classifier = LogisticRegression(max_iter=5000)
    classifier = GridSearchCV(
        classifier,
        param_grid={'C': np.logspace(-4, 4, 9), 'class_weight': ['balanced', None]},
        n_jobs=-1,
        cv=5
    )
    classifier.fit(Xtr, labels)
    classifier = classifier.best_estimator_
    classifier_acc = classifier.best_score_
    print(f'[done] best-params={classifier.best_params_} got {classifier_acc:.4f} score')

    training = LabelledCollection(Xtr, labels)
    print('training classes:', training.classes_)
    print('training prevalence:', training.prevalence())

    return tfidf, classifier


def reduceAtK(data: LabelledCollection, k):
    # if k > len(data):
    #     print(f'[warning] {k=}>{len(data)=}')
    X, y = data.Xy
    X = X[:k]
    y = y[:k]
    return LabelledCollection(X, y, classes=data.classes_)


def benchmark_name(class_name, k):
    scape_class_name = class_name.replace('_', '\_')
    return f'{scape_class_name}@{k}'


def run_experiment():
    results = {
        'mae': {k: [] for k in Ks},
        'mrae': {k: [] for k in Ks}
    }

    pbar = tqdm(experiment_prot(), total=experiment_prot.total())
    for train, test in pbar:
        Xtr, ytr, score_tr = train
        Xte, yte, score_te = test

        if HALF:
            n = len(ytr) // 2
            train_col = LabelledCollection(Xtr[:n], ytr[:n], classes=classifier_trained.classes_)
        else:
            train_col = LabelledCollection(Xtr, ytr, classes=classifier_trained.classes_)

        if method_name not in ['Naive', 'NaiveQuery']:
            quantifier.fit(train_col, val_split=train_col, fit_classifier=False)
        elif method_name == 'Naive':
            quantifier.fit(train_col)

        test_col = LabelledCollection(Xte, yte, classes=classifier_trained.classes_)
        for k in Ks:
            test_k = reduceAtK(test_col, k)
            if method_name == 'NaiveQuery':
                train_k = reduceAtK(train_col, k)
                quantifier.fit(train_k)

            estim_prev = quantifier.quantify(test_k.instances)

            mae = qp.error.mae(test_k.prevalence(), estim_prev)
            mrae = qp.error.mrae(test_k.prevalence(), estim_prev, eps=(1. / (2 * k)))

            results['mae'][k].append(mae)
            results['mrae'][k].append(mrae)

        pbar.set_description(f'{method_name}')

    return results


Ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]

if __name__ == '__main__':
    data_home = 'data'

    HALF=True
    exp_posfix = '_half'

    method_names = [name for name, *other in methods(None, 'continent')]

    for class_name in ['gender', 'continent', 'years_category']: # 'relative_pageviews_category', 'num_sitelinks_category']:
        tables_mae, tables_mrae = [], []

        benchmarks = [benchmark_name(class_name, k) for k in Ks]

        for data_size in ['10K', '50K', '100K', '500K', '1M', 'FULL']:

            table_mae = Table(name=f'{class_name}-{data_size}-mae', benchmarks=benchmarks, methods=method_names)
            table_mrae = Table(name=f'{class_name}-{data_size}-mrae', benchmarks=benchmarks, methods=method_names)
            table_mae.format.mean_prec = 5
            table_mae.format.remove_zero = True
            table_mae.format.color_mode = 'global'

            tables_mae.append(table_mae)
            tables_mrae.append(table_mrae)

            class_home = join(data_home, class_name, data_size)
            # train_data_path = join(class_home, 'classifier_training.json')
            # classifier_path = join('classifiers', data_size, f'classifier_{class_name}.pkl')
            train_data_path = join(data_home, class_name, 'FULL', 'classifier_training.json')  # <-------- fixed classifier
            classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}.pkl')  # <------------ fixed classifier
            test_rankings_path = join(data_home, 'testRanking_Results.json')
            results_home = join('results'+exp_posfix, class_name, data_size)

            tfidf, classifier_trained = qp.util.pickled_resource(classifier_path, train_classifier, train_data_path)

            experiment_prot = RetrievedSamples(
                class_home,
                test_rankings_path,
                vectorizer=tfidf,
                class_name=class_name,
                classes=classifier_trained.classes_
            )
            for method_name, quantifier in methods(classifier_trained, class_name):

                results_path = join(results_home, method_name + '.pkl')
                if os.path.exists(results_path):
                    print(f'Method {method_name=} already computed')
                    results = pickle.load(open(results_path, 'rb'))
                else:
                    results = run_experiment()

                    os.makedirs(Path(results_path).parent, exist_ok=True)
                    pickle.dump(results, open(results_path, 'wb'), pickle.HIGHEST_PROTOCOL)

                for k in Ks:
                    table_mae.add(benchmark=benchmark_name(class_name, k), method=method_name, v=results['mae'][k])
                    table_mrae.add(benchmark=benchmark_name(class_name, k), method=method_name, v=results['mrae'][k])

        # Table.LatexPDF(f'./latex{exp_posfix}/{class_name}{exp_posfix}.pdf', tables=tables_mae+tables_mrae)
        Table.LatexPDF(f'./latex{exp_posfix}/{class_name}{exp_posfix}.pdf', tables=tables_mrae)







