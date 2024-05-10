from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import clone

import quapy as qp
from Retrieval.commons import *
from Retrieval.methods import *
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


def methods(classifier, class_name=None, binarize=False):

    kde_param = {
        'continent': 0.01,
        'gender': 0.03,
        'years_category':0.03
    }

    # yield ('Naive', Naive())
    # yield ('NaiveHalf', Naive())
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
    yield ('KDEy-ML', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=kde_param.get(class_name, 0.01)))
    # yield ('KDE01', KDEyML(classifier, val_split=5, n_jobs=-1, bandwidth=0.01))
    if binarize:
        yield ('M3b', M3rND_ModelB(classifier))
        yield ('M3b+', M3rND_ModelB(classifier))
        yield ('M3d', M3rND_ModelD(classifier))
        yield ('M3d+', M3rND_ModelD(classifier))


def train_classifier_fn(train_path):
    """
    Trains a classifier. To do so, it loads the training set, transforms it into a tfidf representation.
    The classifier is Logistic Regression, with hyperparameters C (range [0.001, 0.01, ..., 1000]) and
    class_weight (range {'balanced', None}) optimized via 5FCV.

    :return: the tfidf-vectorizer and the classifier trained
    """
    texts, labels = load_sample(train_path, class_name=class_name)

    if BINARIZE:
        labels = binarize_labels(labels, positive_class=protected_group[class_name])

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3)
    Xtr = tfidf.fit_transform(texts)
    print(f'Xtr shape={Xtr.shape}')

    print('training classifier...', end='')
    classifier = LogisticRegression(max_iter=5000)
    modsel = GridSearchCV(
        classifier,
        param_grid={'C': np.logspace(-4, 4, 9), 'class_weight': ['balanced', None]},
        n_jobs=-1,
        cv=5
    )
    modsel.fit(Xtr, labels)
    classifier = modsel.best_estimator_
    classifier_acc = modsel.best_score_
    best_params = modsel.best_params_
    print(f'[done] best-params={best_params} got {classifier_acc:.4f} score')

    print('generating cross-val predictions for M3')
    predictions = cross_val_predict(clone(classifier), Xtr, labels, cv=10, n_jobs=-1, verbose=10)
    conf_matrix = confusion_matrix(labels, predictions, labels=classifier.classes_)

    training = LabelledCollection(Xtr, labels)
    print('training classes:', training.classes_)
    print('training prevalence:', training.prevalence())

    return tfidf, classifier, conf_matrix


def reduceAtK(data: LabelledCollection, k):
    # if k > len(data):
    #     print(f'[warning] {k=}>{len(data)=}')
    X, y = data.Xy
    X = X[:k]
    y = y[:k]
    return LabelledCollection(X, y, classes=data.classes_)


def benchmark_name(class_name, k=None):
    scape_class_name = class_name.replace('_', '\_')
    if k is None:
        return scape_class_name
    else:
        return f'{scape_class_name}@{k}'


def run_experiment():

    results = {
        'mae': {k: [] for k in Ks},
        'mrae': {k: [] for k in Ks},
        'rKL_error': [],
        'rND_error': []
    }

    pbar = tqdm(experiment_prot(), total=experiment_prot.total())
    for train, test, q_rel_prevs in pbar:
        Xtr, ytr, score_tr = train
        Xte, yte, score_te = test

        train_col = LabelledCollection(Xtr, ytr, classes=classifier.classes_)

        if not method_name.startswith('Naive') and not method_name.startswith('M3'):
            method.fit(train_col, val_split=train_col, fit_classifier=False)
        elif method_name == 'Naive':
            method.fit(train_col)
        elif method_name == 'NaiveHalf':
            n = len(ytr)//2
            train_col = LabelledCollection(Xtr[:n], ytr[:n], classes=classifier.classes_)
            method.fit(train_col)

        test_col = LabelledCollection(Xte, yte, classes=classifier.classes_)
        rKL_estim, rKL_true = [], []
        rND_estim, rND_true = [], []
        for k in Ks:
            test_k = reduceAtK(test_col, k)
            if method_name == 'NaiveQuery':
                train_k = reduceAtK(train_col, k)
                method.fit(train_k)

            estim_prev = method.quantify(test_k.instances)

            # epsilon value for prevalence smoothing
            eps=(1. / (2. * k))

            # error metrics
            test_k_prev = test_k.prevalence()
            mae = qp.error.mae(test_k_prev, estim_prev)
            mrae = qp.error.mrae(test_k_prev, estim_prev, eps=eps)
            rKL_at_k_estim = qp.error.kld(estim_prev, q_rel_prevs, eps=eps)
            rKL_at_k_true  = qp.error.kld(test_k_prev, q_rel_prevs, eps=eps)

            if BINARIZE:
                # [1] is the index of the minority or historically disadvantaged group
                rND_at_k_estim = np.abs(estim_prev[1] - q_rel_prevs[1])
                rND_at_k_true = np.abs(test_k_prev[1] - q_rel_prevs[1])

            # collect results
            results['mae'][k].append(mae)
            results['mrae'][k].append(mrae)
            rKL_estim.append(rKL_at_k_estim)
            rKL_true.append(rKL_at_k_true)
            if BINARIZE:
                rND_estim.append(rND_at_k_estim)
                rND_true.append(rND_at_k_true)


        # aggregate fairness metrics
        def aggregate(rMs, Ks, Z=1):
            return (1 / Z) * sum((1. / np.log2(k)) * v for v, k in zip(rMs, Ks))

        Z = sum((1. / np.log2(k)) for k in Ks)
        rKL_estim = aggregate(rKL_estim, Ks, Z)
        rKL_true  = aggregate(rKL_true, Ks, Z)
        rKL_error = np.abs(rKL_true-rKL_estim)
        results['rKL_error'].append(rKL_error)

        if BINARIZE:
            rND_estim = aggregate(rND_estim, Ks, Z)
            rND_true = aggregate(rND_true, Ks, Z)

            if isinstance(method, AbstractM3rND):
                if method_name.endswith('+'):
                    # learns the correction parameters from the query-specific training data
                    conf_matrix_ = method.get_confusion_matrix(*train_col.Xy)
                else:
                    # learns the correction parameters from the training data used to train the classifier
                    conf_matrix_ = conf_matrix.copy()
                rND_estim = method.fair_measure_correction(rND_estim, conf_matrix_)

            rND_error = np.abs(rND_true - rND_estim)
            results['rND_error'].append(rND_error)

        pbar.set_description(f'{method_name}')

    return results


data_home = 'data'

if __name__ == '__main__':

    # final tables only contain the information for the data size 10K, each row is a class name and each colum
    # the corresponding rND (for binary) or rKL (for multiclass) score
    tables_RND, tables_DKL = [], []
    tables_final = []
    for class_mode in ['multiclass', 'binary']:
        BINARIZE = (class_mode=='binary')
        method_names = [name for name, *other in methods(None, binarize=BINARIZE)]

        table_final = Table(name=f'rND' if BINARIZE else f'rKL', benchmarks=[benchmark_name(c) for c in CLASS_NAMES], methods=method_names)
        table_final.format.mean_macro = False
        tables_final.append(table_final)
        for class_name in CLASS_NAMES:
            tables_mae, tables_mrae = [], []

            benchmarks_size =[benchmark_name(class_name, s) for s in DATA_SIZES]
            table_DKL = Table(name=f'rKL-{class_name}', benchmarks=benchmarks_size, methods=method_names)
            table_RND = Table(name=f'rND-{class_name}', benchmarks=benchmarks_size, methods=method_names)

            for data_size in DATA_SIZES:
                print(class_name, class_mode, data_size)
                benchmarks_k = [benchmark_name(class_name, k) for k in Ks]
                # table_mae = Table(name=f'{class_name}-{data_size}-mae', benchmarks=benchmarks_k, methods=method_names)
                table_mrae = Table(name=f'{class_name}-{data_size}-mrae', benchmarks=benchmarks_k, methods=method_names)

                # tables_mae.append(table_mae)
                tables_mrae.append(table_mrae)

                # sets all paths
                class_home = join(data_home, class_name, data_size)
                train_data_path = join(data_home, class_name, 'FULL', 'classifier_training.json') # <----- fixed classifier
                classifier_path = join('classifiers', 'FULL', f'classifier_{class_name}_{class_mode}.pkl')
                test_rankings_path = join(data_home, 'testRanking_Results.json')
                test_query_prevs_path = join(data_home, 'prevelance_vectors_judged_docs.json')
                results_home = join('results', class_name, class_mode, data_size)
                positive_class = protected_group[class_name] if BINARIZE else None

                # instantiates the classifier (trains it the first time, loads it in the subsequent executions)
                tfidf, classifier, conf_matrix \
                    = qp.util.pickled_resource(classifier_path, train_classifier_fn, train_data_path)

                experiment_prot = RetrievedSamples(
                    class_home,
                    test_rankings_path,
                    test_query_prevs_path,
                    vectorizer=tfidf,
                    class_name=class_name,
                    positive_class=positive_class,
                    classes=classifier.classes_
                )

                for method_name, method in methods(classifier, class_name, BINARIZE):

                    results_path = join(results_home, method_name + '.pkl')
                    results = qp.util.pickled_resource(results_path, run_experiment)

                    # compose the tables
                    for k in Ks:
                        # table_mae.add(benchmark=benchmark_name(class_name, k), method=method_name, v=results['mae'][k])
                        table_mrae.add(benchmark=benchmark_name(class_name, k), method=method_name, v=results['mrae'][k])
                    table_DKL.add(benchmark=benchmark_name(class_name, data_size), method=method_name, v=results['rKL_error'])
                    if BINARIZE:
                        table_RND.add(benchmark=benchmark_name(class_name, data_size), method=method_name, v=results['rND_error'])

                    if data_size=='10K':
                        value = results['rND_error'] if BINARIZE else results['rKL_error']
                        table_final.add(benchmark=benchmark_name(class_name), method=method_name, v=value)

            tables = ([table_RND] + tables_mrae) if BINARIZE else ([table_DKL] + tables_mrae)
            Table.LatexPDF(f'./latex/{class_mode}/{class_name}.pdf', tables=tables)

            if BINARIZE:
                tables_RND.append(table_RND)
            else:
                tables_DKL.append(table_DKL)

    Table.LatexPDF(f'./latex/global/main.pdf', tables=tables_RND+tables_DKL, dedicated_pages=False)
    Table.LatexPDF(f'./latex/final/main.pdf', tables=tables_final, dedicated_pages=False)







