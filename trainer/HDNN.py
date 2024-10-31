import math
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sample_src = [
    # '../datasets/descriptor_matrix/Au13_descriptor',
    # '../datasets/descriptor_matrix/Au19_descriptor',
    # '../datasets/descriptor_matrix/Au38_descriptor',
    # '../datasets/descriptor_matrix/Au58_descriptor',
    # '../datasets/descriptor_matrix/Au75_descriptor',
    '../datasets/descriptor_matrix/C_descriptor'
]


# def neuralNetwork(name):
#     name = tf.keras.Sequential([tf.keras.layers.Dense(20, input_shape=(10,), activation='relu'),
#                                tf.keras.layers.Dense(1)])
#     return name
class HighDimensionAuClusterNet(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1, epochs=2000, nn_number=1):
        # 网络结构参数设定
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nn_number = nn_number
        # 训练样本集
        # self.train_db, self.test_db, self.val_db = self.load_train_dataset()
        # self.train_batch_length = len(self.train_db)
        # self.test_batch_length = len(self.test_db)
        # self.val_batch_length = len(self.val_db)
        # 前馈神经网络权重参数初始化
        self.w1 = tf.Variable(tf.random.truncated_normal(shape=[self.input_nodes, self.hidden_nodes], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_nodes]))

        self.w2 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_nodes, self.hidden_nodes], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_nodes]))

        self.w3 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_nodes, self.output_nodes], stddev=0.1))
        self.b3 = tf.Variable(tf.random.truncated_normal(shape=[self.output_nodes]))
        # 权重参数列表
        self.train_weight_matrix_list = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        # 优化器指定
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # 训练过程中参数记录
        self.train_batch_loss = 0
        self.train_epoch_loss_list = []

        # 验证集上的损失函数值
        self.val_batch_loss = 0
        self.val_epoch_loss_list = []

    def single_atom_nn_forward(self, descriptor_vector):
        # 计算单原子能量-前向传播过程 26-N-N-1
        descriptor_vector = tf.reshape(descriptor_vector, shape=[1,15])
        hidden1_input = tf.matmul(descriptor_vector, self.w1) + self.b1
        hidden1_output = tf.nn.relu(hidden1_input)
        hidden2_input = tf.matmul(hidden1_output, self.w2) + self.b2
        hidden2_output = tf.nn.relu(hidden2_input)
        output = tf.matmul(hidden2_output, self.w3) + self.b3
        return output

def forward(inputs):
    # 一个batch下的前馈计算
    outputs = []
    # 循环遍历每一个样本矩阵-描述符矩阵
    #1x1650  =====  150 x 11
    # print(inputs)
    inputs = tf.reshape(inputs, shape=[-1, 16])
    # print(inputs)
    energy_value = []
    # 遍历每一个样本下的所有原子数
    Energy_total = 0
    NN = []
    for data in enumerate(inputs):
        # print(data)
        atom = int(data[-1][0:1])
        descriptor_vector = data[-1][1:]
        # print(atom)
        # print(descriptor_vector)
        if np.any(descriptor_vector):
            # 计算单原子对应的原子能
            if atom == 0:
                Energy_C = model_C.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_C
            elif atom == 1:
                Energy_H = model_H.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_H
            elif atom == 2:
                Energy_N = model_N.single_atom_nn_forward(descriptor_vector)
                # print("Energy_N = ",Energy_N)
                Energy_total += Energy_N
            elif atom == 3:
                Energy_O = model_O.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_O
            elif atom == 4:
                Energy_Zr = model_Zr.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_Zr
            elif atom == 5:
                Energy_Au = model_Au.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_Au
            elif atom == 6:
                Energy_Pd = model_Pd.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_Pd
            elif atom == 7:
                Energy_Ir = model_Ir.single_atom_nn_forward(descriptor_vector)
                Energy_total += Energy_Ir
    # print(Energy_total)
    # print(Atom_class)
    # print(outputs)
    # print(Atom_class)
    return Energy_total


def train(train_db, val_db):
    epochs = 2000
    for epoch in range(1, epochs + 1):
        val_batch_loss = 0
        train_batch_loss = 0
        train_db_loss_func = 0
        with tf.GradientTape(persistent=True) as tape:
            for step, (x_train, y_train) in enumerate(train_db):
                for i in range(0,x_train.shape[0]):
                    Atom_class = atom_class(inputs=x_train[i])
                    # print(i)
                    y_pred = forward(inputs=x_train[i])
                    # print(y_pred)
                    # print(Atom_class)
                    # 类型准换
                    y_pred = tf.cast(y_pred, dtype=y_train.dtype)
                    # 损失函数定义
                    loss_func = tf.reduce_mean(tf.square(y_train[i] - y_pred))
                    train_db_loss_func += loss_func
                    train_batch_loss += loss_func.numpy()
        print( '============= train_db_loss_func value is {} ============='.format(train_db_loss_func/len(train_db)))
        for step, (x_train, y_train) in enumerate(train_db):
            for i in range(0, x_train.shape[0]):
                Atom_class = atom_class(inputs=x_train[i])
                for i in Atom_class:
                    if i == 0:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
                        model_C.optimizer(loss='mean_squared_error',optimizer=adam)
                    elif i == 1:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_H.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 2:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_N.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 3:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_O.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 4:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_Zr.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 5:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_Au.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 6:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_Pd.optimizer(loss='mean_squared_error', optimizer=adam)
                    elif i == 7:
                        # 优化器指定
                        adam = optimizers.Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model_Ir.optimizer(loss='mean_squared_error', optimizer=adam)
        # 清空当前batch和损失值
        # 迭代20epoch验证一次数据
        # if epoch % 20 ==0:
        # for _, (x_val, y_val) in enumerate(val_db):
        #     y_pred = forward(inputs=x_val)
        #     y_pred = tf.cast(y_pred, dtype=y_val.dtype)
        #     # y_pred = preprocessing.MinMaxScaler().inverse_transform(y_train)
        #     val_loss_mse = tf.reduce_mean(tf.square(y_val - y_pred))
        #     val_batch_loss += val_loss_mse.numpy()
        #
        # print('Epoch is {}, ============= train loss value is {}.'.format(
        #         epoch, train_batch_loss / len(train_db)
        #     ))
        # print('Epoch is {}, ============= val loss value is {}.'.format(
        #         epoch,val_batch_loss / len(val_db)
        #     ))
        #
        # self.train_batch_loss = 0
        # self.val_batch_loss = 0
        # if save_params and epoch % 10 == 0:
        #     # 指定模型参数保存路径
        #     save_src = '../trainer/checkpoints/nn_%s/epoch_%.4d/' % (self.nn_number, epoch)
        #     if not os.path.exists(save_src):
        #         os.mkdir(save_src)
        #     # 存储训练权重矩阵
        #     filenames = ['w1.npy', 'b1.npy', 'w2.npy', 'b2.npy', 'w3.npy', 'b3.npy']
        #     for idx, weight_matrix in enumerate(self.train_weight_matrix_list):
        #         with open(file=save_src + filenames[idx], mode='wb') as fp:
        #             np.save(fp, weight_matrix.numpy())
def atom_class(inputs):
    inputs = tf.reshape(inputs, shape=[-1, 16])
    NN = []
    for data in enumerate(inputs):
        if np.any(data[-1]):
            atom = int(data[-1][0:1])
            NN.append(atom)
    return NN


if __name__ == '__main__':

    # 训练数据集加载
    model_C = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=0, epochs=50)
    model_H = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=1, epochs=50)
    model_N = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=2, epochs=50)
    model_O = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=3, epochs=50)
    model_Zr = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=4, epochs=50)
    model_Au = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=5, epochs=50)
    model_Pd = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=6, epochs=50)
    model_Ir = HighDimensionAuClusterNet(
        input_nodes=15, hidden_nodes=10, output_nodes=1, learning_rate=0.05, nn_number=7, epochs=50)
    # Au13_sample Au19_sample Au38_sample
    samples = []
    labels = []
    for fold_idx, train_path in enumerate(sample_src):
        for filename in os.listdir(train_path):
            filepath = train_path + '/' + filename
            with open(filepath, mode='rb') as fp:
                # 加载每一个样本个体
                dataset = np.load(fp)
            sample, label = dataset[:, :16], dataset[:, -1][0]
            # 特征数据标准化处理 特征数据处理为[0,1]之间
            for col_idx in range(sample.shape[1]):
                if col_idx == 0:
                    continue
                sample[:, col_idx] = (sample[:, col_idx] - sample[:, col_idx].min()) / (
                        sample[:, col_idx].max() - sample[:, col_idx].min())
            # 数据维度统一
            if sample.shape[0] < 150:
                zero_matrix = np.zeros(shape=[150 - sample.shape[0], 16])
                sample = np.vstack((sample, zero_matrix))
            sample = sample.reshape(1, -1).tolist()[0]
            samples.append(sample)
            labels.append(label)

    samples = np.array(samples)
    labels = np.array(labels).reshape(-1, 1)
    # print(samples)
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.4, shuffle=True,
                                                        random_state=0)
    y_train = preprocessing.MinMaxScaler().fit_transform(y_train)
    y_test = preprocessing.MinMaxScaler().fit_transform(y_test)

    x_val = x_test[: 50, :]
    y_val = y_test[:50, :]
    # print(type(x_train))
    # print(type(x_test))
    # 样本数据和标签数据转换为Tensor格式
    print('训练集样本大小： {}'.format(x_train.shape[0]), '验证集样本大小：{}'.format(x_val.shape[0]))

    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    x_val = tf.cast(x_val, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.float32)
    y_val = tf.cast(y_val, dtype=tf.float32)
    # 数据批次打包 一共300条数据，打包成40 batch
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1) #30 x 1650
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=1)
    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size=1)
    # print(model_C.train_weight_matrix_list)
    train(train_db,val_db)
    # test_NN = []
    # Y_test = []
    # for _, (x_test, y_test) in enumerate(test_db):
    #     y_NN = forward(inputs=x_test)
    #     y_NN = tf.cast(y_NN, dtype=y_val.dtype)
    #     test_NN.append(y_NN)
    #     Y_test.append(y_test)
    # plt.figure()
    # plt.plot(test_NN, 'r', marker='*', label="HDNN")
    # plt.plot(Y_test, 'b', marker='o', label="DFT")
    # plt.legend(loc="best")
    # plt.show()
