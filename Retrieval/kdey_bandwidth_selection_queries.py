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
from experiments import benchmark_name, reduceAtK, run_experiment

from os.path import join
from tqdm import tqdm

from result_table.src.table import Table



def methods(classifier):
    for i, bandwidth in enumerate(np.linspace(0.01, 0.1, 10)):
        yield (f'KDE{str(i).zfill(2)}', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=bandwidth))


if __name__ == '__main__':
    data_home = 'data-modsel'

    Ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]

    method_names = [m for m, *_ in methods(None)]

    class_mode = 'multiclass'

    dir_names={
        'gender': '100K_GENDER_TREC21_QUERIES/100K-NEW-QUERIES',
        'continent': '100K_CONT_TREC21_QUERIES/100K-NEW-QUERIES',
        'years_category': '100K_YEARS_TREC21_QUERIES/100K-NEW-QUERIES'
    }

    for class_name in ['gender', 'continent', 'years_category']:

        tables_mrae = []

        benchmarks = [benchmark_name(class_name, k) for k in Ks]

        for data_size in ['100K']:

            table_mrae = Table(name=f'{class_name}-{data_size}-mrae', benchmarks=benchmarks, methods=method_names)
            tables_mrae.append(table_mrae)

            class_home = join(data_home, dir_names[class_name])
            classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}_{class_mode}.pkl')
            test_rankings_path = join(data_home, 'testRanking-TREC21-Queries_Results.json')
            test_query_prevs_path = join('data', 'prevelance_vectors_judged_docs.json')
            results_home = join('results', 'modsel', class_name, data_size)

            tfidf, classifier, conf_matrix = pickle.load(open(classifier_path, 'rb'))

            experiment_prot = RetrievedSamples(
                class_home,
                test_rankings_path,
                test_query_prevs_path,
                vectorizer=tfidf,
                class_name=class_name,
                classes=classifier.classes_
            )
            for method_name, quantifier in methods(classifier):

                results_path = join(results_home, method_name + '.pkl')
                results = qp.util.pickled_resource(results_path, run_experiment)

                for k in Ks:
                    table_mrae.add(benchmark=benchmark_name(class_name, k), method=method_name, v=results['mrae'][k])

                Table.LatexPDF(f'./latex/modsel/{class_name}.pdf', tables=tables_mrae)







