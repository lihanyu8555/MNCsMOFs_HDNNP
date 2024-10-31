# -*- coding = utf-8 -*-
# @Author：风城浪子
# @File：training.py
# @Time：2023/2/22 20:50
# @Software：PyCharm

"""
    TensorFlow 训练 Au13_sample Au19_sample Au38_sample, Au58_sample, Au75_sample
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HighDimensionAuClusterNet(object):
    sample_src = [
        # '../datasets/descriptor_matrix/Au13_descriptor',
        # '../datasets/descriptor_matrix/Au19_descriptor',
        # '../datasets/descriptor_matrix/Au38_descriptor',
        # '../datasets/descriptor_matrix/Au58_descriptor',
        # '../datasets/descriptor_matrix/Au75_descriptor',
        '../datasets/descriptor_matrix/C_descriptor'
    ]
    max_dimension = 150
    # 单原子结合能
    single_atom_energy = -0.28316556

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1, epochs=2000, nn_number=1):
        # 网络结构参数设定
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nn_number = nn_number
        # 训练样本集
        self.train_db, self.test_db, self.val_db = self.load_train_dataset()
        self.train_batch_length = len(self.train_db)
        self.test_batch_length = len(self.test_db)
        self.val_batch_length = len(self.val_db)
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

    def load_train_dataset(self):
        """训练数据集加载"""
        samples = []
        labels = []
        # Au13_sample Au19_sample Au38_sample
        for fold_idx, train_path in enumerate(self.sample_src):
            for filename in os.listdir(train_path):
                filepath = train_path + '/' + filename
                with open(filepath, mode='rb') as fp:
                    # 加载每一个样本个体
                    dataset = np.load(fp)
                sample, label = dataset[:, :26], dataset[:, -1][0]
                # # 能量数据标准化
                label = (label - sample.shape[0] * self.single_atom_energy) / sample.shape[0]
                # 特征数据标准化处理 特征数据处理为[0,1]之间
                for col_idx in range(sample.shape[1]):
                    sample[:, col_idx] = (sample[:, col_idx] - sample[:, col_idx].min()) / (
                            sample[:, col_idx].max() - sample[:, col_idx].min())
                # 数据维度统一
                if sample.shape[0] < self.max_dimension:
                    zero_matrix = np.zeros(shape=[self.max_dimension - sample.shape[0], 26])
                    sample = np.vstack((sample, zero_matrix))
                sample = sample.reshape(1, -1).tolist()[0]
                samples.append(sample)
                labels.append(label)
        # N * 988 numpy格式
        samples = np.array(samples)
        labels = np.array(labels).reshape(-1, 1)
        # print(samples.shape) # (3000, 988)
        # todo 数据划分
        x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.9, shuffle=True,
                                                            random_state=0)
        x_val = x_test[: 50, :]
        y_val = y_test[:50, :]
        # 样本数据和标签数据转换为Tensor格式
        print('训练集样本大小： {}'.format(x_train.shape[0]), '验证集样本大小：{}'.format(x_val.shape[0]))

        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        x_val = tf.cast(x_val, dtype=tf.float32)
        y_train = tf.cast(y_train, dtype=tf.float32)
        y_test = tf.cast(y_test, dtype=tf.float32)
        y_val = tf.cast(y_val, dtype=tf.float32)
        # 数据批次打包 一共3000条数据，打包成40 batch
        train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=32)
        val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size=32)
        return train_db, test_db, val_db

    def single_atom_nn_forward(self, descriptor_vector):
        # 计算单原子能量-前向传播过程 26-N-N-1
        descriptor_vector = tf.reshape(descriptor_vector, shape=[1, self.input_nodes])
        hidden1_input = tf.matmul(descriptor_vector, self.w1) + self.b1
        hidden1_output = tf.nn.relu(hidden1_input)
        hidden2_input = tf.matmul(hidden1_output, self.w2) + self.b2
        hidden2_output = tf.nn.relu(hidden2_input)
        output = tf.matmul(hidden2_output, self.w3) + self.b3
        return output

    def forward(self, inputs):
        # 一个batch下的前馈计算
        outputs = []
        # 循环遍历每一个样本矩阵-描述符矩阵
        for _, sample_matrix in enumerate(inputs):
            sample_matrix = tf.reshape(sample_matrix, shape=[-1, self.input_nodes])
            energy_value = []
            # 遍历每一个样本下的所有原子数
            for _, descriptor_vector in enumerate(sample_matrix):

                if np.any(descriptor_vector):
                    # 计算单原子对应的原子能
                    output = self.single_atom_nn_forward(descriptor_vector=descriptor_vector)
                    energy_value.append(output)
            # 计算每个样本的平均原子能
            outputs.append(tf.reduce_mean(energy_value))
        outputs = tf.cast(outputs, dtype=tf.float32)
        return outputs

    def train(self, save_params=True):
        for epoch in range(1, self.epochs + 1):
            for step, (x_train, y_train) in enumerate(self.train_db):
                with tf.GradientTape() as tape:
                    y_pred = self.forward(inputs=x_train)
                    # 类型准换
                    y_pred = tf.cast(y_pred, dtype=y_train.dtype)
                    # 损失函数定义
                    loss_func = tf.reduce_mean(tf.square(y_train - y_pred))
                    # 损失函数值记录
                    self.train_batch_loss += loss_func.numpy()
                # 损失函数对其权重求取梯度
                gradients = tape.gradient(target=loss_func, sources=self.train_weight_matrix_list)
                # 梯度更新
                self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.train_weight_matrix_list))

            # 清空当前batch和损失值
            # 迭代20epoch验证一次数据
            for _, (x_val, y_val) in enumerate(self.val_db):
                y_pred = self.forward(inputs=x_val)
                y_pred = tf.cast(y_pred, dtype=y_val.dtype)
                val_loss_mse = tf.reduce_mean(tf.square(y_val - y_pred))
                self.val_batch_loss += val_loss_mse.numpy()

            print('Epoch is {}, ============= val loss value is {}.'.format(
                epoch,  self.val_batch_loss / self.val_batch_length
            ))

            self.train_batch_loss = 0
            self.val_batch_loss = 0
            if save_params and epoch % 10 == 0:
                # 指定模型参数保存路径
                save_src = '../trainer/checkpoints/nn_%s/epoch_%.4d/' % (self.nn_number, epoch)
                if not os.path.exists(save_src):
                    os.mkdir(save_src)
                # 存储训练权重矩阵
                filenames = ['w1.npy', 'b1.npy', 'w2.npy', 'b2.npy', 'w3.npy', 'b3.npy']
                for idx, weight_matrix in enumerate(self.train_weight_matrix_list):
                    with open(file=save_src + filenames[idx], mode='wb') as fp:
                        np.save(fp, weight_matrix.numpy())


if __name__ == '__main__':
    model = HighDimensionAuClusterNet(
        input_nodes=10, hidden_nodes=25, output_nodes=1, learning_rate=0.05, nn_number=1, epochs=50
    )
    model.load_train_dataset()

    model.train(save_params=False)
