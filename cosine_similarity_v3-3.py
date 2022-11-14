import pandas as pd
import ast
import numpy as np
import torch
import datetime



ROW_START = 140
ROW_END = 160
COL_START = 1
COL_END = 259

ROW_PATH ='data/'
COL_PATH = 'data/'
DEST_PATH = 'features/'


def pairwise_cosine_sim(M, N):
    if torch.has_cudnn:
        M = M.cuda()
        N = N.cuda()
    batch_size = M.size(0)
    M_num_users = M.size(1)
    N_num_users = N.size(1)
    emb_size = M.size(2)

    final = torch.zeros(batch_size, M_num_users, N_num_users)
    if torch.has_cudnn:
        final = final.cuda()

    for i in range(batch_size):
        final[i] = torch.mm(M[i], N[i].transpose(0, 1))
        normalizer = torch.mm(M[i].norm(dim=1, p=2.0).unsqueeze(1), N[i].norm(dim=1, p=2.0).unsqueeze(0))
        final[i] /= normalizer

    return final


zero_row = [0] * 50


def zero_pad(target):
    max_length = max(len(row) for row in target)
    for row in target:
        row += [zero_row] * (max_length - len(row))
    return target




def calculate_similarity_user_i(user, user_info, result):
    k = 0
    for user_j in timeline_tweets_column:

        if user_j == 'Unnamed: 0':
            continue
        # if (user, user_j) in already_calculated and (user, user_j) in already_calculated_similarities.keys():
        #     print('**here**')
        #     result.at[user, user_j] = already_calculated_similarities[(user, user_j)]
        #     continue
        if user == user_j or (user, user_j) in already_calculated:
            # result.at[user, user_j] = 0.0
            continue

        # k += 1
        # print(k)

        user_i_tweets_time = user_info[0]
        user_j_tweets_time = column_data[user_j]['time']
        timestamps = set(user_i_tweets_time).intersection(set(user_j_tweets_time))

        user_i_tweets_text = user_info[1]
        user_j_tweets_text = column_data[user_j]['text']

        num = 0
        similarity = 0
        ms = []
        ns = []
        for ts in timestamps:
            target_i = [user_i_tweets_text[i] for i, x in enumerate(user_i_tweets_time) if x == ts]
            target_j = [user_j_tweets_text[j] for j, x in enumerate(user_j_tweets_time) if x == ts]
            num += len(target_i) * len(target_j)
            ms.append(target_i)
            ns.append(target_j)

        if len(timestamps):
            ms = torch.tensor(zero_pad(ms))
            ns = torch.tensor(zero_pad(ns))
            similarity = pairwise_cosine_sim(ms, ns)
            similarity[similarity != similarity] = 0
            similarity = similarity.sum()

        if num != 0 and similarity != 0:
            similarity = similarity / num
            result.at[user, user_j] = similarity

        already_calculated.add((user_j, user))
        # already_calculated_similarities[(user_j, user)] = similarity


for row in range(ROW_START, ROW_END):
    timeline_tweets_row = pd.read_csv(ROW_PATH + 'timeline_tweets' + str(row) + '.csv')
    row_data = {}
    for user in timeline_tweets_row:
        if user == 'Unnamed: 0':
            continue
        row_data[user] = {'text': ast.literal_eval(timeline_tweets_row[user][1]),
                          'time': ast.literal_eval(timeline_tweets_row[user][0])}

    print('prepared row')
    for column in range(COL_START, COL_END):
        begin_time = datetime.datetime.now()
        print('row: ', row)
        print('column: ', column)
        timeline_tweets_column = pd.read_csv(COL_PATH + 'timeline_tweets' + str(column) + '.csv')
        column_data = {}
        for user in timeline_tweets_column:
            if user == 'Unnamed: 0':
                continue
            column_data[user] = {'text': ast.literal_eval(timeline_tweets_column[user][1]),
                                 'time': ast.literal_eval(timeline_tweets_column[user][0])}

        print('prepared column')

        already_calculated = set()
        # already_calculated_similarities = {}

        columns = timeline_tweets_column.columns.values
        columns = columns[columns != 'Unnamed: 0']
        base = {}
        for c in columns:
            base[c] = [0.0]

        c = 0
        for user_i in timeline_tweets_row:

            if user_i == 'Unnamed: 0':
                continue

            print('round ' + str(c))
            print(str(c * 100 / len(columns)) + '%')

            features = base
            features['user'] = user_i
            features = pd.DataFrame.from_dict(features)
            features.set_index('user', inplace=True)

            user_i_tweets_time = row_data[user_i]['time']
            user_i_tweets_text = row_data[user_i]['text']

            calculate_similarity_user_i(user_i, (user_i_tweets_time, user_i_tweets_text), features)

            if c == 0:
                features.to_csv(DEST_PATH + str(row) + '-' + str(column) + '.csv', float_format='%.5f')
                print(features)
            else:
                features.to_csv(DEST_PATH + str(row) + '-' + str(column) + '.csv', mode='a', header=False,
                                float_format='%.5f')
                print(features)

            c += 1

        print(datetime.datetime.now() - begin_time)


