import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_path = '/home/wgq/research/bs/cnn/model/32x32/test6500.log'

def cal_th(df, th):
    df[3] = df[0] < th  # ns
    df[4] = df[3] == df[2]

    ns = df[df[3] == 0]
    ns_correct = ns[ns[4] == True]
    num_ns = ns.shape[0]
    num_ns_correct = ns_correct.shape[0]

    s = df[df[3] == 1]
    s_correct = s[s[4] == True]
    num_s = s.shape[0]
    num_s_correct = s_correct.shape[0]

    return num_ns, num_ns_correct, num_s, num_s_correct
    # if num_ns == 0:
    #     print("s|ns: %.4f(%d/%d)"%(0, 0, 0))
    # else:
    #     print("s|ns: %.4f(%d/%d)"%(1- num_ns_correct/num_ns, num_ns - num_ns_correct, num_ns))
    # if num_s == 0:
    #     print("ns|s: %.4f(%d/%d)"%(0, 0, 0))
    # else:
    #     print("ns|s: %.4f(%d/%d)"%(1- num_s_correct/num_s, num_s - num_s_correct, num_s))


df = pd.read_csv(data_path, header=None)
risk = pd.DataFrame(columns=['th', 'ns', 'ns_correct', 's', 's_correct'])
num = 0
for i in np.linspace(0, 1.05, 22):
    # print(i)
    ns, ns_correct, s, s_correct = cal_th(df, i)
    risk.loc[num, 'th'] = i
    risk.loc[num, 'ns'] = ns
    risk.loc[num, 'ns_correct'] = ns_correct
    risk.loc[num, 's'] = s
    risk.loc[num, 's_correct'] = s_correct
    num = num + 1
# print(risk)
risk['s_ns'] = 1 - risk['ns_correct']/risk.loc[risk['ns']!= 0, 'ns']
risk['ns_s'] = 1 - risk['s_correct']/risk.loc[risk['s']!= 0, 's']
risk = risk.replace(np.nan, 0)
print(risk)
risk.to_csv(data_path.split('.')[0] + '.csv')
# df[3] = df[df[0] < df[1]]

# print(df)