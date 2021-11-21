import pandas as pd
import os

data_path = '/home/wgq/research/bs/dataset/allintra'
csv_list = [f for f in os.listdir(data_path)]

for csv_file in csv_list:

    if 'train' in csv_file:
        print(csv_file)
        df = pd.read_csv(data_path + '/' + csv_file, header=None)
        df00 = df[(df[3]==0) & (df[4]==0)]
        df01 = df[(df[3]==0) & (df[4]==1)]
        df10 = df[(df[3]==1) & (df[4]==0)]
        df11 = df[(df[3]==1) & (df[4]==1)]
        length = min(df00.shape[0], df01.shape[0], df10.shape[0], df11.shape[0])
        if length > 20000:
            length = 20000
        df00 = df00.sample(n=length, replace=False, axis=0)
        df01 = df01.sample(n=length, replace=False, axis=0)
        df10 = df10.sample(n=length, replace=False, axis=0)
        df11 = df11.sample(n=length, replace=False, axis=0)
        df = pd.concat([df00, df01, df10, df11], axis=0)
        df.to_csv(data_path + '/t' + csv_file, header=0, index=0)

        # break

# train_list.close()
# df.to_csv(data_path + '1', header=0, index=0)