import pandas as pd

from os.path import join

from Retrieval.commons import load_json_sample
from quapy.data import LabelledCollection

data_home = 'data'
CLASS_NAME = 'continent'
datasize = '100K'

file_path = join(data_home, CLASS_NAME, datasize, 'training_Query-84Sample-200SPLIT.json')

text, classes = load_json_sample(file_path, CLASS_NAME)


data = LabelledCollection(text, classes)
print(data.classes_)
print(data.prevalence())
print('done')

test_ranking_path = join(data_home, 'testRanking_Results.json')
# obj = json.load(open(test_ranking_path))


df = pd.read_json(test_ranking_path)
print('done')