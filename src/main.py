import tensorflow as tf
import os
import numpy as np
import pandas as pd
from src.BiLSTM import BiLSTM

column_num = ['date', 'serial_number', 'model', 'failure', 'capacity_bytes', 'smart_1_normalized',
              'smart_2_normalized', 'smart_3_normalized', 'smart_4_normalized', 'smart_5_normalized',
              'smart_7_normalized', 'smart_8_normalized', 'smart_9_normalized', 'smart_10_normalized',
              'smart_12_normalized', 'smart_183_normalized', 'smart_184_normalized', 'smart_187_normalized',
              'smart_188_normalized', 'smart_189_normalized', 'smart_190_normalized', 'smart_191_normalized',
              'smart_192_normalized', 'smart_193_normalized', 'smart_194_normalized', 'smart_195_normalized',
              'smart_196_normalized', 'smart_197_normalized', 'smart_198_normalized', 'smart_199_normalized',
              'smart_235_normalized', 'smart_240_normalized', 'smart_241_normalized', 'smart_242_normalized']

if __name__ == '__main__':
    with tf.Session() as sess:

        model = BiLSTM(step_size=90, in_size=29, out_size=1, hidden_layer=128, batch_size=100, learning_rate=0.5)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data = pd.read_csv('../data/total.csv')
        index = pd.read_csv('../data/index.csv')['serial_number']
        index_num = 0
        for num1 in range(812):
            data_in = []
            label_in = []
            fail = 0
            for num2 in range(100):
                temp = data[data['serial_number'] == index[index_num]]
                for n in temp.loc[:, 'failure'].values:
                    label_in.append([n])
                    if n ==1:
                        fail+=1
                data_in.append(np.array(
                    temp.drop(['failure', 'serial_number', 'date', 'model', 'capacity_bytes'], axis=1)).tolist())
            index_num += 1
            if num1 == 0:
                # 初始化 data
                feed_dict = {
                    model.xs: data_in,
                    model.ys: label_in,
                }
            else:
                feed_dict = {
                    model.xs: data_in,
                    model.ys: label_in,
                    model.fw_init_state: fw_state,
                    model.bw_init_state: bw_state  # 保持 state 的连续性
                }

                # 训练
            _, loss, fw_state, bw_state, pred = sess.run(
                [model.train, model.loss, model.fw_final_state, model.bw_final_state, model.output],
                feed_dict=feed_dict)

            weight = sess.run(model.weight)
            biases = sess.run(model.biases)
            print("------------------------------------------------------")
            print(pred)
            # print("after epoch %d, the loss is %6f" % (num1, loss))
            print("------------------------------------------------------")

            # print("the in_W is %f, the in_b is %f,the out_W is %f, the out_b is %f " % (
            #     sess.run(weight['in']), sess.run(biases['in']), sess.run(weight['out']),
            #     sess.run(biases['out'])))
            if num1 % 100 == 0:
                saver.save(sess, "../data/model/my-model", global_step=num1)
