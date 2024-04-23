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



def methods(classifier, class_name):
    yield ('KDE001', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.001))
    yield ('KDE005', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.005))
    yield ('KDE01', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.01))
    yield ('KDE02', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.02))
    yield ('KDE03', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.03))
    yield ('KDE04', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.04))
    yield ('KDE05', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.05))
    yield ('KDE07', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.07))
    yield ('KDE10', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.10))

def reduceAtK(data: LabelledCollection, k):
    # if k > len(data):
    #     print(f'[warning] {k=}>{len(data)=}')
    X, y = data.Xy
    X = X[:k]
    y = y[:k]
    return LabelledCollection(X, y, classes=data.classes_)


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

def benchmark_name(class_name, k):
    scape_class_name = class_name.replace('_', '\_')
    return f'{scape_class_name}@{k}'


if __name__ == '__main__':
    data_home = 'data-modsel'

    HALF=True
    exp_posfix = '_half_modsel'

    Ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]

    method_names = [m for m, *_ in methods(None, None)]

    dir_names={
        'gender': '100K_GENDER_TREC21_QUERIES/100K-NEW-QUERIES',
        'continent': '100K_CONT_TREC21_QUERIES/100K-NEW-QUERIES',
        'years_category': '100K_YEARS_TREC21_QUERIES/100K-NEW-QUERIES'
    }

    for class_name in ['gender', 'continent', 'years_category']: # 'relative_pageviews_category', 'num_sitelinks_category']:
        tables_mae, tables_mrae = [], []

        benchmarks = [benchmark_name(class_name, k) for k in Ks]

        for data_size in ['100K']:

            table_mae = Table(name=f'{class_name}-{data_size}-mae', benchmarks=benchmarks, methods=method_names)
            table_mrae = Table(name=f'{class_name}-{data_size}-mrae', benchmarks=benchmarks, methods=method_names)
            table_mae.format.mean_prec = 5
            table_mae.format.remove_zero = True
            table_mae.format.color_mode = 'global'

            tables_mae.append(table_mae)
            tables_mrae.append(table_mrae)

            class_home = join(data_home, dir_names[class_name])
            classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}.pkl')  # <------------ fixed classifier
            test_rankings_path = join(data_home, 'testRanking-TREC21-Queries_Results.json')
            results_home = join('results'+exp_posfix, class_name, data_size)

            tfidf, classifier_trained = pickle.load(open(classifier_path, 'rb'))

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







