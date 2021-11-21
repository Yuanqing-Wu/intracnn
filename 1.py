import pandas as pd
import os

data_path = '/home/wgq/research/bs/dataset/allintra/test_list_64x64.txt'
df = pd.read_csv(data_path, header=None)
print(df)
df[4] = 0
df.to_csv(data_path + '1', header=0, index=0)