import os
import pandas as pd


def train_data_combine():
    train_path = '../data/data_Q1_2018'
    train_file_names = None
    for root, dirs, files in os.walk(train_path):
        train_file_names = files.copy()
    column_num = ['date', 'serial_number', 'model', 'failure', 'capacity_bytes', 'smart_1_normalized',
                  'smart_2_normalized', 'smart_3_normalized', 'smart_4_normalized', 'smart_5_normalized',
                  'smart_7_normalized', 'smart_8_normalized', 'smart_9_normalized', 'smart_10_normalized',
                  'smart_12_normalized', 'smart_183_normalized', 'smart_184_normalized', 'smart_187_normalized',
                  'smart_188_normalized', 'smart_189_normalized', 'smart_190_normalized', 'smart_191_normalized',
                  'smart_192_normalized', 'smart_193_normalized', 'smart_194_normalized', 'smart_195_normalized',
                  'smart_196_normalized', 'smart_197_normalized', 'smart_198_normalized', 'smart_199_normalized',
                  'smart_235_normalized', 'smart_240_normalized', 'smart_241_normalized', 'smart_242_normalized']
    data = None
    for name in train_file_names:
        temp = pd.read_csv(train_path + '/' + name, usecols=column_num)
        temp = temp.fillna(0.0)
        if name is train_file_names[0]:
            data = temp.copy()
        else:
            data = data.append(temp)
    data.to_csv('../data/total.csv', index=False)
    print('train data combine is over!')


def index_pro():
    train_path = '../data/data_Q1_2018'
    train_file_names = None
    # serial_num = pd.read_csv('../data/index.csv')['serial_number']
    for root, dirs, files in os.walk(train_path):
        train_file_names = files.copy()
    column_num = ['serial_number', 'model']
    data = None
    for name in train_file_names:
        temp = pd.read_csv(train_path + '/' + name, usecols=column_num)
        if name == train_file_names[0]:
            data = temp
        else:
            data = data.merge(temp, on='serial_number')
    data.to_csv('index.csv', index=False)
    print('over')


index_pro()
# train_data_combine()
# train_data_generate()
