from copy import deepcopy
from typing import Union
from tqdm import tqdm

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

import quapy as qp
from quapy.data import LabelledCollection
from quapy import functional as F
from quapy.evaluation import evaluate
from quapy.model_selection import GridSearchQ
from . import neural
from .base import BaseQuantifier
from quapy.method.aggregative import CC, ACC, PCC, PACC, HDy, EMQ

QuaNet = neural.QuaNetTrainer


class Ensemble(BaseQuantifier):

    VALID_POLICIES = {'ave', 'ptr', 'ds'} | qp.error.QUANTIFICATION_ERROR_NAMES

    """
    Methods from the articles:
    Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
    Using ensembles for problems with characterizable changes in data distribution: A case study on quantification.
    Information Fusion, 34, 87-100.
    and
    Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019). 
    Dynamic ensemble selection for quantification tasks. 
    Information Fusion, 45, 1-15.
    """

    def __init__(self, quantifier: BaseQuantifier, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1, verbose=False):
        assert policy in Ensemble.VALID_POLICIES, f'unknown policy={policy}; valid are {Ensemble.VALID_POLICIES}'
        self.base_quantifier = quantifier
        self.size = size
        self.min_pos = min_pos
        self.red_size = red_size
        self.policy = policy
        self.n_jobs = n_jobs
        self.post_proba_fn = None
        self.verbose = verbose

    def sout(self, msg):
        if self.verbose:
            print('[Ensemble]' + msg)

    def fit(self, data: qp.data.LabelledCollection, val_split:Union[qp.data.LabelledCollection, float]=None):
        self.sout('Fit')
        if self.policy=='ds' and not data.binary:
            raise ValueError(f'ds policy is only defined for binary quantification, but this dataset is not binary')

        # randomly chooses the prevalences for each member of the ensemble (preventing classes with less than
        # min_pos positive examples)
        prevs = [_draw_simplex(ndim=data.n_classes, min_val=self.min_pos / len(data)) for _ in range(self.size)]

        posteriors = None
        if self.policy == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.post_proba_fn = self.ds_policy_get_posteriors(data)

        is_static_policy = (self.policy in qp.error.QUANTIFICATION_ERROR_NAMES)
        self.ensemble = Parallel(n_jobs=self.n_jobs)(
            delayed(_delayed_new_instance)(
                self.base_quantifier, data, val_split, prev, posteriors, keep_samples=is_static_policy, verbose=self.verbose
            ) for prev in tqdm(prevs, desc='fitting ensamble')
        )

        # static selection policy (the name of a quantification-oriented error function to minimize)
        if self.policy in qp.error.QUANTIFICATION_ERROR_NAMES:
            self.accuracy_policy(error_name=self.policy)

        self.sout('Fit [Done]')
        return self

    def quantify(self, instances):
        predictions = np.asarray(Parallel(n_jobs=self.n_jobs)(
            delayed(_delayed_quantify)(Qi, instances) for Qi in self.ensemble
        ))

        if self.policy == 'ptr':
            predictions = self.ptr_policy(predictions)
        elif self.policy == 'ds':
            predictions = self.ds_policy(predictions, instances)

        predictions = np.mean(predictions, axis=0)
        return F.normalize_prevalence(predictions)

    def set_params(self, **parameters):
        raise NotImplementedError(f'{self.__class__.__name__} should not be used within GridSearchQ; '
                                  f'instead, use Ensemble(GridSearchQ(q),...), with q a Quantifier (recommended), '
                                  f'or Ensemble(Q(GridSearchCV(l))) with Q a quantifier class that has a learner '
                                  f'l optimized for classification (not recommended).')

    def get_params(self, deep=True):
        raise NotImplementedError()

    def accuracy_policy(self, error_name):
        """
        Selects the red_size best performant quantifiers in a static way (i.e., dropping all non-selected instances).
        For each model in the ensemble, the performance is measured in terms of _error_name_ on the quantification of
        the samples used for training the rest of the models in the ensemble.
        """
        error = getattr(qp.error, error_name)
        tests = [m[3] for m in self.ensemble]
        scores = []
        for i, model in enumerate(self.ensemble):
            scores.append(evaluate(model[0], tests[:i] + tests[i+1:], error, self.n_jobs))
        order = np.argsort(scores)

        self.ensemble = select_k(self.ensemble, order, k=self.red_size)

    def ptr_policy(self, predictions):
        """
        Selects the predictions made by models that have been trained on samples with a prevalence that is most similar
        to a first approximation of the test prevalence as made by all models in the ensemble.
        """
        test_prev_estim = predictions.mean(axis=0)
        tr_prevs = [m[1] for m in self.ensemble]
        ptr_differences = [qp.error.mse(ptr_i, test_prev_estim) for ptr_i in tr_prevs]
        order = np.argsort(ptr_differences)
        return select_k(predictions, order, k=self.red_size)

    def ds_policy_get_posteriors(self, data: LabelledCollection):
        """
        In the original article, this procedure is not described in a sufficient level of detail. The paper only says
        that the distribution of posterior probabilities from training and test examples is compared by means of the
        Hellinger Distance. However, how these posterior probabilities are generated is not specified. In the article,
        a Logistic Regressor (LR) is used as the classifier device and that could be used for this purpose. However, in
        general, a Quantifier is not necessarily an instance of Aggreggative Probabilistic Quantifiers, and so that the
        quantifier builds on top of a probabilistic classifier cannot be given for granted. Additionally, it would not
        be correct to generate the posterior probabilities for training documents that have concurred in training the
        classifier that generates them.
        This function thus generates the posterior probabilities for all training documents in a cross-validation way,
        using a LR with hyperparameters that have previously been optimized via grid search in 5FCV.
        :return P,f, where P is a ndarray containing the posterior probabilities of the training data, generated via
        cross-validation and using an optimized LR, and the function to be used in order to generate posterior
        probabilities for test instances.
        """
        X, y = data.Xy
        lr_base = LogisticRegression(class_weight='balanced', max_iter=1000)

        optim = GridSearchCV(
            lr_base, param_grid={'C': np.logspace(-4,4,9)}, cv=5, n_jobs=self.n_jobs, refit=True
        ).fit(X, y)

        posteriors = cross_val_predict(
            optim.best_estimator_, X, y, cv=5, n_jobs=self.n_jobs, method='predict_proba'
        )
        posteriors_generator = optim.best_estimator_.predict_proba

        return posteriors, posteriors_generator

    def ds_policy(self, predictions, test):
        test_posteriors = self.post_proba_fn(test)
        test_distribution = get_probability_distribution(test_posteriors)
        tr_distributions = [m[2] for m in self.ensemble]
        dist = [F.HellingerDistance(tr_dist_i, test_distribution) for tr_dist_i in tr_distributions]
        order = np.argsort(dist)
        return select_k(predictions, order, k=self.red_size)

    @property
    def binary(self):
        return self.base_quantifier.binary

    @property
    def aggregative(self):
        return False
        #raise NotImplementedError('aggregative functionality not yet supported for Ensemble')

    @property
    def probabilistic(self):
        return False
        #raise NotImplementedError('probabilistic functionality not yet supported for Ensemble')
        #return self.base_quantifier.probabilistic


def get_probability_distribution(posterior_probabilities, bins=8):
    assert posterior_probabilities.shape[1]==2, 'the posterior probabilities do not seem to be for a binary problem'
    posterior_probabilities = posterior_probabilities[:,1]  # take the positive posteriors only
    distribution, _ = np.histogram(posterior_probabilities, bins=bins, range=(0, 1), density=True)
    return distribution


def select_k(elements, order, k):
    return [elements[idx] for idx in order[:k]]


def _delayed_new_instance(base_quantifier,
                          data: LabelledCollection,
                          val_split: Union[LabelledCollection, float],
                          prev,
                          posteriors,
                          keep_samples,
                          verbose):
    if verbose:
        print(f'\tfit-start for prev {F.strprev(prev)}')
    model = deepcopy(base_quantifier)
    sample_index = data.sampling_index(len(data), *prev)
    sample = data.sampling_from_index(sample_index)
    if val_split is None:
        model.fit(sample)
    else:
        if isinstance(val_split, float):
            assert 0<val_split<1, 'val_split should be in (0,1)'
            sample, val_split = sample.split_stratified(train_prop=1-val_split)
        model.fit(sample, val_split=val_split)
    tr_prevalence = sample.prevalence()
    tr_distribution = get_probability_distribution(posteriors[sample_index]) if (posteriors is not None) else None
    if verbose:
        print(f'\t\--fit-ended for prev {F.strprev(prev)}')
    return (model, tr_prevalence, tr_distribution, sample if keep_samples else None)


def _delayed_quantify(quantifier, instances):
    return quantifier[0].quantify(instances)


def _draw_simplex(ndim, min_val, max_trials=100):
    """
    returns a uniform sampling from the ndim-dimensional simplex but guarantees that all dimensions
    are >= min_class_prev (for min_val>0, this makes the sampling not truly uniform)
    :param ndim: number of dimensions of the simplex
    :param min_val: minimum class prevalence allowed. If less than 1/ndim a ValueError will be throw since
    there is no possible solution.
    :return: a sample from the ndim-dimensional simplex that is uniform in S(ndim)-R where S(ndim) is the simplex
    and R is the simplex subset containing dimensions lower than min_val
    """
    if min_val >= 1/ndim:
        raise ValueError(f'no sample can be draw from the {ndim}-dimensional simplex so that '
                         f'all its values are >={min_val} (try with a larger value for min_pos)')
    trials = 0
    while True:
        u = F.uniform_simplex_sampling(ndim)
        if all(u >= min_val):
            return u
        trials += 1
        if trials >= max_trials:
            raise ValueError(f'it looks like finding a random simplex with all its dimensions being'
                             f'>= {min_val} is unlikely (it failed after {max_trials} trials)')


def _instantiate_ensemble(learner, base_quantifier_class, param_grid, optim, param_model_sel, **kwargs):
    if optim is None:
        base_quantifier = base_quantifier_class(learner)
    elif optim in qp.error.CLASSIFICATION_ERROR:
        learner = GridSearchCV(learner, param_grid)
        base_quantifier = base_quantifier_class(learner)
    elif optim in qp.error.QUANTIFICATION_ERROR:
        base_quantifier = GridSearchQ(base_quantifier_class(learner),
                                      param_grid=param_grid,
                                      **param_model_sel,
                                      error=optim)
    else:
        raise ValueError(f'value optim={optim} not understood')

    return Ensemble(base_quantifier, **kwargs)


def _check_error(error):
    if error is None:
        return None
    if error in qp.error.QUANTIFICATION_ERROR or error in qp.error.CLASSIFICATION_ERROR:
        return error
    elif isinstance(error, str):
        assert error in qp.error.ERROR_NAMES, \
            f'unknown error name; valid ones are {qp.error.ERROR_NAMES}'
        return getattr(qp.error, error)
    else:
        raise ValueError(f'unexpected error type; must either be a callable function or a str representing\n'
                         f'the name of an error function in {qp.error.ERROR_NAMES}')


def ensembleFactory(learner, base_quantifier_class, param_grid=None, optim=None,
                 param_model_sel:dict=None,
                 size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
        if optim is not None:
            if param_grid is None:
                raise ValueError(f'param_grid is None but optim was requested.')
            if param_model_sel is None:
                raise ValueError(f'param_model_sel is None but optim was requested.')
        error = _check_error(optim)
        return _instantiate_ensemble(learner, base_quantifier_class, param_grid, error, param_model_sel,
                                     size=size, min_pos=min_pos, red_size=red_size,
                                     policy=policy, n_jobs=n_jobs)


def ECC(learner, param_grid=None, optim=None, param_mod_sel=None, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
    return ensembleFactory(learner, CC, param_grid, optim, param_mod_sel, size, min_pos, red_size, policy, n_jobs)


def EACC(learner, param_grid=None, optim=None, param_mod_sel=None, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
    return ensembleFactory(learner, ACC, param_grid, optim, param_mod_sel, size, min_pos, red_size, policy, n_jobs)


def EPACC(learner, param_grid=None, optim=None, param_mod_sel=None, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
    return ensembleFactory(learner, PACC, param_grid, optim, param_mod_sel, size, min_pos, red_size, policy, n_jobs)


def EHDy(learner, param_grid=None, optim=None, param_mod_sel=None, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
    return ensembleFactory(learner, HDy, param_grid, optim, param_mod_sel, size, min_pos, red_size, policy, n_jobs)


def EEMQ(learner, param_grid=None, optim=None, param_mod_sel=None, size=50, min_pos=1, red_size=25, policy='ave', n_jobs=1):
    return ensembleFactory(learner, EMQ, param_grid, optim, param_mod_sel, size, min_pos, red_size, policy, n_jobs)