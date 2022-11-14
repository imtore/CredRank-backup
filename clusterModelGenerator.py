import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
from scipy.sparse import hstack, csr_matrix, triu, coo_matrix, vstack
import scipy
from joblib import dump, load
import pickle
from sklearn.decomposition import IncrementalPCA

batch_size = 1000

ROW_START = 1
ROW_END = 259
COL_START = 1
COL_END = 259

users = []
models = {}


transformer = IncrementalPCA(n_components=2, batch_size=1000)

ns = [30,40,50]

for i in ns:
    models[i] = MiniBatchKMeans(n_clusters=i, random_state=10, batch_size=1000)

rows = list(range(ROW_START, ROW_END))
remaining_rows = 257105

while remaining_rows > 0:
    if len(rows) > 0:
        row = rows.pop(0)
        start = datetime.datetime.now()
        cframes = []
        print('row: ' + str(row))
        for col in range(1, row):
            similarities = pd.read_csv(
                'cosineSimilarity/features/row' + str(col) + '/' + str(col) + '-' + str(row) + '.csv', index_col=0,
                dtype={'0': str}).transpose()
            cframes.append(similarities.values)
            if row == 1:
                users = users + list(similarities.columns.values)
        for col in range(row, COL_END):
            similarities = pd.read_csv(
                'cosineSimilarity/features/row' + str(row) + '/' + str(row) + '-' + str(col) + '.csv', index_col=0,
                dtype={'0': str})
            cframes.append(similarities.values)
            if row == 1:
                users = users + list(similarities.columns.values)

        print('reading all row\'s columns done')

        row_arr = np.concatenate(cframes, axis=1)

        if row == 1:
            features = row_arr

        else:
            features = np.concatenate([features, row_arr], axis=0)


        print('concatenating row done')

        print(datetime.datetime.now() - start)

        if features.shape[0] >= 1000:
            transformer.partial_fit(row_arr[0:batch_size, :])
            for i in ns:
                models[i] = models[i].partial_fit(row_arr[0:batch_size, :])
            features = features[:batch_size, :]
            remaining_rows -= batch_size
        else:
            transformer.partial_fit(row_arr[0:len(features), :])
            for i in ns:
                models[i] = models[i].partial_fit(row_arr[0:len(features), :])
            remaining_rows -= len(features)

print('collecting rows done')

dump(transformer, 'transformer.joblib')
for i in ns:
    dump(models[i], 'minibatchkm' + str(i) + 'c.joblib')

