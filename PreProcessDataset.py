import pandas as pd
import os

data_path = '/home/wgq/research/bs/dataset/32x32'
csv_list = [f for f in os.listdir(data_path)]

train_list = open(data_path + '/test_list.txt', 'w')
for csv_file in csv_list:
    if csv_file.endswith('.csv'):
        print(csv_file)
        df = pd.read_csv(data_path + '/' + csv_file, header=None)
        df = df[df[0] == 0]

        pos = df[df[6]==df[12]]  # 不划分
        neg = df[df[6]!=df[12]]  # 划分
        #print(pos)

        if pos.shape[0] < neg.shape[0]:
            neg = neg.sample(n=int(len(pos)), replace=False, axis=0)
        if pos.shape[0] > neg.shape[0]:
            pos = pos.sample(n=int(len(neg)), replace=False, axis=0)

        for i in pos.index:
            train_list.write(csv_file.split('.csv')[0] + ',' + '%08d'%(i) + ',' + str(df.iloc[i, 5]) + ',0\n') # 不划分
        for i in neg.index:
            train_list.write(csv_file.split('.csv')[0] + ',' + '%08d'%(i) + ',' + str(df.iloc[i, 5]) + ',1\n') #划分

        # break

train_list.close()