import pandas as pd

from os.path import join

from quapy.data import LabelledCollection

data_home = 'data'
CLASS_NAME = 'continent'
datasize = '100K'

file_path = join(data_home, 'prevelance_vectors_judged_docs.json')

df = pd.read_json(file_path)

pd.set_option('display.max_columns', None)
print(df)