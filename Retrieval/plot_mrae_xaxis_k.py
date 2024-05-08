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
from Retrieval.experiments import methods, benchmark_name
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as Naive
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC, KDEyML
from quapy.data.base import LabelledCollection

from os.path import join
from tqdm import tqdm

from result_table.src.table import Table
import matplotlib.pyplot as plt



data_home = 'data'
class_mode = 'multiclass'

method_names = [name for name, *other in methods(None, 'continent')]

# Ks = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]
Ks = [50, 100, 500, 1000]
DATA_SIZE = ['10K', '50K', '100K', '500K', '1M', 'FULL']
CLASS_NAME = ['gender', 'continent', 'years_category']
all_results = {}


# loads all MRAE results, and returns a dictionary containing the values, which is indexed by:
# class_name -> data_size -> method_name -> k -> stat -> float
# where stat is "mean", "std", "max"
def load_all_results():
    for class_name in CLASS_NAME:

        all_results[class_name] = {}

        for data_size in DATA_SIZE:

            all_results[class_name][data_size] = {}

            results_home = join('results', class_name, class_mode, data_size)

            all_results[class_name][data_size] = {}

            for method_name in method_names:
                results_path = join(results_home, method_name + '.pkl')
                try:
                    results = pickle.load(open(results_path, 'rb'))
                except Exception as e:
                    print(f'missing result {results}', e)

                all_results[class_name][data_size][method_name] = {}
                for k in Ks:
                    all_results[class_name][data_size][method_name][k] = {}
                    values = results['mrae']
                    all_results[class_name][data_size][method_name][k]['mean'] = np.mean(values[k])
                    all_results[class_name][data_size][method_name][k]['std'] = np.std(values[k])
                    all_results[class_name][data_size][method_name][k]['max'] = np.max(values[k])

    return all_results


results = load_all_results()

# generates the class-independent, size-independent plots for y-axis=MRAE in which:
# - the x-axis displays the Ks

for class_name in CLASS_NAME:
    for data_size in DATA_SIZE:

        fig, ax = plt.subplots()

        max_means = []
        for method_name in method_names:
            # class_name -> data_size -> method_name -> k -> stat -> float
            means = [
                results[class_name][data_size][method_name][k]['mean'] for k in Ks
            ]
            stds = [
                results[class_name][data_size][method_name][k]['std'] for k in Ks
            ]
            # max_mean = np.max([
            #         results[class_name][data_size][method_name][k]['max'] for k in Ks
            # ])
            max_means.append(max(means))

            means = np.asarray(means)
            stds = np.asarray(stds)

            line = ax.plot(Ks, means, 'o-', label=method_name, color=None)
            color = line[-1].get_color()
            # ax.fill_between(Ks, means - stds, means + stds, alpha=0.3, color=color)

        ax.set_xlabel('k')
        ax.set_ylabel('RAE')
        ax.set_title(f'{class_name} from {data_size}')
        ax.set_ylim([0, max(max_means)*1.05])

        ax.legend()

        os.makedirs(f'plots/var_k/{class_name}', exist_ok=True)
        plotpath = f'plots/var_k/{class_name}/{data_size}_mrae.pdf'
        print(f'saving plot in {plotpath}')
        plt.savefig(plotpath, bbox_inches='tight')











