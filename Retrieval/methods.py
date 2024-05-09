
"""
This file implements some of the methods presented in the FAccT'22 paper by
Ghazimatin, Kleindessner, Russell, Abedjan, and Golebiowski,
Measuring Fairness of Rankings under Noisy Sensitive Information.

In particular, it implements two variants of a method relying on M3=rND:
one in which the assumed graphical model is P(Â,A,S) = P(Â|A)*P(S|A) (called "b")
and another in which the assumed graphical model is P(Â,A,S) = P(Â|A)*P(S|Â) (called "d")
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix

from quapy.method.aggregative import CC


class AbstractM3rND(ABC):
    def __init__(self, classifier):
        self.quantifier = CC(classifier)

    def proxy_labels(self, instances):
        return self.quantifier.classify(instances)

    def quantify(self, instances):
        return self.quantifier.quantify(instances)

    @abstractmethod
    def fair_measure_correction(self, rND_estim: float, conf_matrix: np.ndarray):
        ...

    def get_confusion_matrix(self, X, y, additive_smoothing=0.5):
        """
        Some confusion matrices may contain 0 values for certain classes, and this causes
        instabilities in the correction. If requested, applies additive smoothing. Default
        is adding half a count.

        :param X: array-like with the covariates
        :param y: array-like with the true labels
        :param additive_smoothing: float, default 0.5
        :return: the confusion matrix C with entries Cij=P(Y=i,Ŷ=j)
        """
        proxy_labels = self.proxy_labels(X)
        true_labels = y
        labels = self.quantifier.classes_
        conf_matrix = confusion_matrix(true_labels, proxy_labels, labels=labels)
        if additive_smoothing > 0:
            conf_matrix = conf_matrix.astype(float) + additive_smoothing
        return conf_matrix


class M3rND_ModelB(AbstractM3rND):
    def __init__(self, classifier):
        super().__init__(classifier)

    def fair_measure_correction(self, rND_estim: float, conf_matrix: np.ndarray):
        # conf_matrix contains values Cij=P(Y=i,Ŷ=j)
        # truecond_matrix contains values Cij=P(Ŷ=j|Y=i) (truecond stands for "conditioned on true labels")
        truecond_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        p = truecond_matrix[0, 1]  # P(hat{A}=1|A=0)
        q = truecond_matrix[1, 0]  # P(hat{A}=0|A=1)
        den = (1 - p - q)
        if den != 0:
            corr = 1./den
            rND_estim = rND_estim * corr
        return rND_estim


class M3rND_ModelD(AbstractM3rND):
    def __init__(self, classifier):
        super().__init__(classifier)

    def fair_measure_correction(self, rND_estim: float, conf_matrix: np.ndarray):
        # conf_matrix contains values Cij=P(Y=i,Ŷ=j)
        # truecond_matrix contains values Cij=P(Ŷ=j|Y=i) (truecond stands for "conditioned on true labels")
        truecond_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        prev_A = conf_matrix.sum(axis=1)
        beta = prev_A[1]  # P(A)
        p = truecond_matrix[0, 1]  # P(hat{A}=1|A=0)
        q = truecond_matrix[1, 0]  # P(hat{A}=0|A=1)
        x = (1 - q) * beta + p * (1 - beta)
        y = q * beta + (1 - p) * (1 - beta)
        if x != 0 and y != 0:
            corr = ((((1 - q) * beta) / x) - (q * beta / y))
            rND_estim = rND_estim * corr
        return rND_estim

