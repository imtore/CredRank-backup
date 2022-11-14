from joblib import load
import datetime
import pandas as pd
import numpy as np
import pickle
import math

ROW_START = 1
ROW_END = 259
COL_START = 1
COL_END = 259

transformer = load('transformer.joblib')
models = {}

X = []

# range_n_clusters = range(13, 23)
range_n_clusters = [30,40,50]
user_cluster_labels = {}

for n_clusters in range_n_clusters:
    models[n_clusters] = load('minibatchkm' + str(n_clusters) + 'c.joblib')
    user_cluster_labels[n_clusters] = {}


for row in range(ROW_START, ROW_END):
    start = datetime.datetime.now()
    cframes = []
    row_users = []
    print('row: ' + str(row))
    for col in range(1, row):
        similarities = pd.read_csv(
            'cosineSimilarity/features/row' + str(col) + '/' + str(col) + '-' + str(row) + '.csv', index_col=0,
            dtype={'0': str}).transpose()

        cframes.append(similarities.values)
        if col == 1:
            row_users = list(similarities.index)
    for col in range(row, COL_END):
        similarities = pd.read_csv(
            'cosineSimilarity/features/row' + str(row) + '/' + str(row) + '-' + str(col) + '.csv', index_col=0,
            dtype={'0': str})

        cframes.append(similarities.values)
        if col == 1:
            row_users = list(similarities.index)
    print('reading all row\'s columns done')

    row_arr = np.concatenate(cframes, axis=1)
    print(row_arr.shape)
    print('concatenating row done')

    partial_x = transformer.transform(row_arr)
    if row == 1:
        X = partial_x
    else:
        X = np.concatenate([X, partial_x], axis=0)


    for n_clusters in range_n_clusters:
        row_labels = models[n_clusters].predict(row_arr)
        for i in range(len(row_users)):
            user_cluster_labels[n_clusters][row_users[i]] = row_labels[i]

        print(len(user_cluster_labels[n_clusters]))

    print(datetime.datetime.now() - start)

print(len(user_cluster_labels))
# print('HERE2')

for n_clusters in range_n_clusters:
    with open('user_cluster_labels' + str(n_clusters) + '.pkl', 'wb') as f:
        pickle.dump(user_cluster_labels[n_clusters], f)

with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)

for n_clusters in range_n_clusters:
    total_num_members = len(user_cluster_labels[n_clusters])
    cluster_mem_count = [0] * n_clusters

    for i in user_cluster_labels[n_clusters].keys():
        label = user_cluster_labels[n_clusters][i]
        cluster_mem_count[label] += 1

    users_list = []
    scores_list = []

    cluster_mem_sqrt = [math.sqrt(cluster_mem_count[i]) for i in range(n_clusters)]
    for i in user_cluster_labels[n_clusters].keys():
        l = user_cluster_labels[n_clusters][i]
        cluster_weight = cluster_mem_sqrt[l] / sum(cluster_mem_count)
        member_weight = cluster_mem_sqrt[l] / cluster_mem_count[l]
        users_list.append(i)
        scores_list.append(member_weight)

    result = pd.DataFrame(data=scores_list, index=users_list)
    print(result)
    result.to_csv('politifact_credrank_scores'+str(n_clusters)+'.csv', float_format='%.5f')
