import pandas as pd
import os

data_path = '/home/wgq/research/bs/VVCSoftware_VTM/video'
csv_list = [f for f in os.listdir(data_path)]

#train_list = open(data_path + '/test_list.txt', 'w')
def save_list(filepath, df, w, h):

    train_list = open(data_path + '/test_list_' + str(w) + 'x' + str(h) + '.txt', 'a')

    df = df[((df[3] == w) & (df[4] == h)) | ((df[3] == w) & (df[4] == h))]

    if w == 64 or w == 4 or h == 4:
        pos = df[df[6]==df[12]]  # 不划分
        neg = df[df[6]!=df[12]]  # 划分

        # if pos.shape[0] < neg.shape[0]:
        #     neg = neg.sample(n=int(len(pos)), replace=False, axis=0)
        # if pos.shape[0] > neg.shape[0]:
        #     pos = pos.sample(n=int(len(neg)), replace=False, axis=0)

        for i in pos.index:
            train_list.write(filepath + ',' + '%d'%(pos.loc[i, 13]) + ',' + str(pos.loc[i, 5]) + ',0\n') # 不划分
        for i in neg.index:
            train_list.write(filepath + ',' + '%d'%(neg.loc[i, 13]) + ',' + str(neg.loc[i, 5]) + ',1\n') #划分
    else:
        valid = df[(df[8]!=-1) | (df[9]!=-1) | (df[10]!=-1) | (df[11]!=-1)]

        pos = valid[valid[6]==valid[12]]  # 不划分
        neg = valid[valid[6]!=valid[12]]  # 划分

        # if pos.shape[0] < neg.shape[0]:
        #     neg = neg.sample(n=int(len(pos)), replace=False, axis=0)
        # if pos.shape[0] > neg.shape[0]:
        #     pos = pos.sample(n=int(len(neg)), replace=False, axis=0)

        bt_tt_pos = pos.iloc[:, [8, 9, 10, 11]]
        h_pos = bt_tt_pos[((bt_tt_pos[8] == bt_tt_pos.max(axis=1)) & (bt_tt_pos[8] !=-1)) | ((bt_tt_pos[10] == bt_tt_pos.max(axis=1)) & (bt_tt_pos[10] !=-1))]
        v_pos = bt_tt_pos[((bt_tt_pos[9] == bt_tt_pos.max(axis=1)) & (bt_tt_pos[9] !=-1)) | ((bt_tt_pos[11] == bt_tt_pos.max(axis=1)) & (bt_tt_pos[11] !=-1))]

        bt_tt_neg = neg.iloc[:, [8, 9, 10, 11]]
        h_neg = bt_tt_neg[((bt_tt_neg[8] == bt_tt_neg.max(axis=1)) & (bt_tt_neg[8] !=-1)) | ((bt_tt_neg[10] == bt_tt_neg.max(axis=1)) & (bt_tt_neg[10] !=-1))]
        v_neg = bt_tt_neg[((bt_tt_neg[9] == bt_tt_neg.max(axis=1)) & (bt_tt_neg[9] !=-1)) | ((bt_tt_neg[11] == bt_tt_neg.max(axis=1)) & (bt_tt_neg[11] !=-1))]

        # print(df)
        for i in h_pos.index:
            train_list.write(filepath + ',' + '%d'%(df.loc[i, 13]) + ',' + str(df.loc[i, 5]) + ',0,0\n') # 不划分
        for i in v_pos.index:
            train_list.write(filepath + ',' + '%d'%(df.loc[i, 13]) + ',' + str(df.loc[i, 5]) + ',0,1\n') # 不划分
        for i in h_neg.index:
            train_list.write(filepath + ',' + '%d'%(df.loc[i, 13]) + ',' + str(df.loc[i, 5]) + ',1,0\n') # 不划分
        for i in v_neg.index:
            train_list.write(filepath + ',' + '%d'%(df.loc[i, 13]) + ',' + str(df.loc[i, 5]) + ',1,1\n') # 不划分
    train_list.close()


for csv_file in csv_list:

    if csv_file.endswith('.csv'):
        print(csv_file)
        pos = 0
        df = pd.read_csv(data_path + '/' + csv_file, header=None)
        df[13] = 0
        for i in range(1, df.shape[0]):
            w, h = df.iloc[i-1, 3], df.iloc[i-1, 4]
            pos = pos + w*h
            df.iloc[i, 13] = pos

        save_list(csv_file.split('.')[0], df, 64, 64)
        df.to_csv(data_path + '1', header=0, index=0)
        # save_list(csv_file.split('.')[0], df, 32, 32)
        # save_list(csv_file.split('.')[0], df, 16, 16)
        # save_list(csv_file.split('.')[0], df, 8, 8)
        # save_list(csv_file.split('.')[0], df, 32, 16)
        # save_list(csv_file.split('.')[0], df, 32, 8)
        # save_list(csv_file.split('.')[0], df, 16, 8)
        df.to_csv(data_path + '/' + csv_file + '1', header=0, index=0)
        # break

# train_list.close()
# df.to_csv(data_path + '1', header=0, index=0)