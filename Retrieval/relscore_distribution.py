import os.path
import pickle
from itertools import zip_longest
from Retrieval.commons import RetrievedSamples, load_sample, DATA_SIZES
from os.path import join
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

"""
Plots the distribution of (predicted) relevance score for the test samples and for the training samples wrt:
- training pool size (100K, 500K, 1M, FULL)
- rank  
"""


data_home = 'data'

for class_name in ['num_sitelinks_category', 'relative_pageviews_category', 'years_category', 'continent', 'gender']:
    test_added = False
    Mtrs, Mtes, source = [], [], []
    for data_size in DATA_SIZES:

        class_home = join(data_home, class_name, data_size)
        classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}.pkl')
        test_rankings_path = join(data_home, 'testRanking_Results.json')

        _, classifier = pickle.load(open(classifier_path, 'rb'))

        experiment_prot = RetrievedSamples(
            class_home,
            test_rankings_path,
            vectorizer=None,
            class_name=class_name,
            classes=classifier.classes_
        )

        Mtr = []
        Mte = []
        pbar = tqdm(experiment_prot(), total=experiment_prot.total())
        for train, test in pbar:
            Xtr, ytr, score_tr = train
            Xte, yte, score_te = test
            Mtr.append(score_tr)
            Mte.append(score_te)

        Mtrs.append(Mtr)
        if not test_added:
            Mtes.append(Mte)
            test_added = True
        source.append(data_size)

    fig, ax = plt.subplots()
    train_source = ['train-'+s for s in source]
    Ms = list(zip(Mtrs, train_source))+list(zip(Mtes, ['test']))

    for M, source in Ms:
        M = np.asarray(list(zip_longest(*M, fillvalue=np.nan))).T

        num_rep, num_docs = M.shape

        mean_values = np.nanmean(M, axis=0)
        n_filled = np.count_nonzero(~np.isnan(M), axis=0)
        std_errors = np.nanstd(M, axis=0) / np.sqrt(n_filled)

        line = ax.plot(range(num_docs), mean_values, '-', label=source, color=None)
        color = line[-1].get_color()
        ax.fill_between(range(num_docs), mean_values - std_errors, mean_values + std_errors, alpha=0.3, color=color)


    ax.set_xlabel('Doc. Rank')
    ax.set_ylabel('Rel. Score')
    ax.set_title(class_name)

    ax.legend()

    # plt.show()
    os.makedirs('plots', exist_ok=True)
    plotpath = f'plots/{class_name}.pdf'
    print(f'saving plot in {plotpath}')
    plt.savefig(plotpath)



