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
from Retrieval.experiments import methods
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as Naive
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.data.base import LabelledCollection

from os.path import join
from tqdm import tqdm

from result_table.src.table import Table
import matplotlib.pyplot as plt


def benchmark_name(class_name, k):
    scape_class_name = class_name.replace('_', '\_')
    return f'{scape_class_name}@{k}'


data_home = 'data'

HALF=True
exp_posfix = '_half'

method_names = [name for name, *other in methods(None, 'continent')]

Ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]

for class_name in ['gender', 'continent', 'years_category']: # 'relative_pageviews_category', 'num_sitelinks_category']:

    benchmarks = [benchmark_name(class_name, k) for k in Ks]

    for data_size in ['10K', '50K', '100K', '500K', '1M', 'FULL']:

        fig, ax = plt.subplots()

        class_home = join(data_home, class_name, data_size)
        test_rankings_path = join(data_home, 'testRanking_Results.json')
        results_home = join('results'+exp_posfix, class_name, data_size)

        max_mean = None
        for method_name in method_names:

            results_path = join(results_home, method_name + '.pkl')
            try:
                results = pickle.load(open(results_path, 'rb'))
            except Exception as e:
                print(f'missing result {results}', e)

            for err in ['mrae']:
                means, stds = [], []
                for k in Ks:
                    values = results[err][k]
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                means = np.asarray(means)
                stds = np.asarray(stds) #/ np.sqrt(len(stds))

                if max_mean is None:
                    max_mean = np.max(means)
                else:
                    max_mean = max(max_mean, np.max(means))

                line = ax.plot(Ks, means, 'o-', label=method_name, color=None)
                color = line[-1].get_color()
                # ax.fill_between(Ks, means - stds, means + stds, alpha=0.3, color=color)

        ax.set_xlabel('k')
        ax.set_ylabel(err.upper())
        ax.set_title(f'{class_name} from {data_size}')
        ax.set_ylim([0, max_mean])

        ax.legend()

        # plt.show()
        os.makedirs(f'plots/results/{class_name}', exist_ok=True)
        plotpath = f'plots/results/{class_name}/{data_size}_{err}.pdf'
        print(f'saving plot in {plotpath}')
        plt.savefig(plotpath)











