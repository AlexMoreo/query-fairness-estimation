import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import quapy as qp
import quapy.functional as F
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.method.aggregative import ClassifyAndCount, EMQ, ACC, PCC, PACC
from quapy.protocol import AbstractProtocol
from quapy.data.base import LabelledCollection

from glob import glob
from os.path import join

"""
This was the very first experiment. 1 big training set and many test rankings produced according to some queries.
The quantification methods did not seem to work. The more sophisticated the method is, the worse it performed.
This is a clear indication that the PPS assumptions do not hold.
Actually, while the training set could be some iid sample from a distribution L and every test set
is a iid sample from a distribution U, it is pretty clear that P(X|Y) is different, since the test set
are biased towards a query term whereas the training set is not.  
"""

def methods():
    yield ('MLPE', MaximumLikelihoodPrevalenceEstimation())
    yield ('CC', ClassifyAndCount(LogisticRegression(n_jobs=-1)))
    yield ('ACC', ACC(LogisticRegression(n_jobs=-1)))
    yield ('PCC', PCC(LogisticRegression(n_jobs=-1)))
    yield ('PACC', PACC(LogisticRegression(n_jobs=-1)))
    yield ('EMQ', EMQ(LogisticRegression(n_jobs=-1)))


def load_txt_sample(path, verbose=False):
    if verbose:
        print(f'loading {path}...', end='')
    df = pd.read_csv(path, sep='\t')
    if verbose:
        print('[done]')
    X = df['text']
    y = df['first_letter_category']

    return X, y

class RetrievedSamples(AbstractProtocol):

    def __init__(self, path_dir: str, load_fn, vectorizer, classes):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.vectorizer = vectorizer
        self.classes = classes

    def __call__(self):
        for file in glob(join(self.path_dir, 'test_data_*.txt')):
            X, y = self.load_fn(file)
            if len(X)!=qp.environ['SAMPLE_SIZE']:
                print(f'[warning]: file {file} contains {len(X)} instances (expected: {qp.environ["SAMPLE_SIZE"]})')
            # assert len(X) == qp.environ['SAMPLE_SIZE'], f'unexpected sample size for file {file}, found {len(X)}'
            X = self.vectorizer.transform(X)
            sample = LabelledCollection(X, y, classes=self.classes)
            yield sample.Xp


qp.environ['SAMPLE_SIZE']=100

data_path = './data'
train_path = join(data_path, 'train_data.txt')


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5)

training = LabelledCollection.load(train_path, loader_func=load_txt_sample, verbose=True)

# training = training.sampling(1000)

Xtr, ytr = training.Xy
Xtr = tfidf.fit_transform(Xtr)
print('Xtr shape = ', Xtr.shape)

training = LabelledCollection(Xtr, ytr)
classes = training.classes_

test_prot = RetrievedSamples(data_path, load_fn=load_txt_sample, vectorizer=tfidf, classes=classes)

print('Training prevalence:', F.strprev(training.prevalence()))
for X, p in test_prot():
    print('Test prevalence:', F.strprev(p))

for method_name, quantifier in methods():
    print('training ', method_name)
    quantifier.fit(training)
    print('[done]')

    report = qp.evaluation.evaluation_report(quantifier, test_prot, error_metrics=['mae', 'mrae'], verbose=True)

    print(report.mean())



