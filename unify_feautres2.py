import pandas as pd
import sys

inp = int(sys.argv[1])

ROW_START = inp
ROW_END = inp - 1
COL_START = 1
COL_END = 259

frames = []
for row in range(ROW_START, ROW_END, -1):
    print('row: '+ str(row))
    for col in range(1, row):
        similarities = pd.read_csv('features/row' + str(col) + '/' + str(col) + '-' + str(row) + '.csv', index_col=0, dtype={'0': str}).transpose()
        frames.append(similarities)
    for col in range(row, COL_END):
        # print('col: '+ str(col))
        similarities = pd.read_csv('features/row' + str(row) + '/' + str(row) + '-' + str(col) + '.csv', index_col=0, dtype={'0': str})
        # print(len(list(similarities.index)))
        frames.append(similarities)

    print('reading data done')

    user_similarities_df = pd.concat(frames, axis=1)

    print('done concatenating')


    user_similarities_df.to_parquet('unified features/'+'row'+str(row)+'.parquet')
