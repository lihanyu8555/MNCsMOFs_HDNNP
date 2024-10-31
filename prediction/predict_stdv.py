# -*- coding = utf-8 -*-
# @Author：风城浪子
# @File：predict_stdv.py
# @Time：2023/2/22 21:08
# @Software：PyCharm

import os
import numpy as np
import tensorflow as tf
from descriptor.descriptor_func import AtomNeighborDescriptor


class StructureFilter(object):
    sample_src = [
        '../datasets/xyz_coordinate/Au13_xyz',
        '../datasets/xyz_coordinate/Au19_xyz',
        '../datasets/xyz_coordinate/Au38_xyz',
        # '../datasets/xyz_coordinate/Au58_xyz',
        # '../datasets/xyz_coordinate/Au75_xyz',
    ]
    max_dimension = 38

    # 单原子结合能
    # single_atom_energy = -0.28316556
    def load_sample_dataset(self):
        samples = []
        labels = []
        for fold_idx, src in enumerate(self.sample_src):
            for idx, filename in enumerate(os.listdir(src)):
                filepath = src + '/' + filename
                with open(file=filepath, mode='r', encoding='utf-8') as fr:
                    lines = fr.readlines()
                # atoms = int(lines[0].strip())
                energy_value = float(lines[1].strip().split('=')[-1].strip('au').strip())
                labels.append(energy_value)
                coordinate_matrix = []
                for line in lines[2:]:
                    line = line.strip().split(' ')[1:]
                    coordinate_matrix.append([float(coordinate) for coordinate in line if coordinate.strip()])
                coordinate_matrix = np.array(coordinate_matrix)
                if coordinate_matrix.shape[0] < self.max_dimension:
                    zero_matrix = np.zeros(shape=[self.max_dimension - coordinate_matrix.shape[0], 3])
                    coordinate_matrix = np.vstack((coordinate_matrix, zero_matrix))
                coordinate_matrix = coordinate_matrix.reshape(1, -1)
                samples.append(coordinate_matrix.tolist()[0])
        samples = np.array(samples)
        labels = np.array(labels).reshape(-1, 1)
        return samples, labels

    def __init__(self):
        self.samples, self.labels = self.load_sample_dataset()
        self.descriptor = AtomNeighborDescriptor()

    def predict_energy(self, coordinate_matrix, w1, b1, w2, b2, w3, b3):
        # coordinate_matrix m * 3
        radial_descriptor_matrix = self.descriptor.retrieve_radial_descriptor_matrix(
            coordinate_matrix=coordinate_matrix)
        angle_descriptor_matrix = self.descriptor.retrieve_angle_descriptor_matrix(coordinate_matrix=coordinate_matrix)
        descriptor_matrix = np.hstack((radial_descriptor_matrix, angle_descriptor_matrix))
        predict_value = []
        descriptor_matrix = tf.cast(descriptor_matrix, dtype=tf.float32)
        # m * 26 matrix
        for _, descriptor_vector in enumerate(descriptor_matrix):
            # 计算单原子能量-前向传播过程 26-N-N-1
            descriptor_vector = tf.reshape(descriptor_vector, shape=[1, 26])
            hidden1_input = tf.matmul(descriptor_vector, w1) + b1
            hidden1_output = tf.nn.relu(hidden1_input)
            hidden2_input = tf.matmul(hidden1_output, w2) + b2
            hidden2_output = tf.nn.relu(hidden2_input)
            output = tf.matmul(hidden2_output, w3) + b3
            predict_value.append(output)
        predict_value = tf.cast(predict_value, dtype=tf.float32)
        return tf.reduce_mean(predict_value)

    # 样本筛选部分
    def filter_sample(self):
        model_weight_matrix_src = {
            'nn_1': [
                {'w1': '../trainer/checkpoints/nn_1/epoch_0050/w1.npy'},
                {'b1': '../trainer/checkpoints/nn_1/epoch_0050/b1.npy'},
                {'w2': '../trainer/checkpoints/nn_1/epoch_0050/w2.npy'},
                {'b2': '../trainer/checkpoints/nn_1/epoch_0050/b2.npy'},
                {'w3': '../trainer/checkpoints/nn_1/epoch_0050/w3.npy'},
                {'b3': '../trainer/checkpoints/nn_1/epoch_0050/b3.npy'},
            ],
            'nn_2': [
                {'w1': '../trainer/checkpoints/nn_2/epoch_0050/w1.npy'},
                {'b1': '../trainer/checkpoints/nn_2/epoch_0050/b1.npy'},
                {'w2': '../trainer/checkpoints/nn_2/epoch_0050/w2.npy'},
                {'b2': '../trainer/checkpoints/nn_2/epoch_0050/b2.npy'},
                {'w3': '../trainer/checkpoints/nn_2/epoch_0050/w3.npy'},
                {'b3': '../trainer/checkpoints/nn_2/epoch_0050/b3.npy'},
            ],
            'nn_3': [
                {'w1': '../trainer/checkpoints/nn_3/epoch_0050/w1.npy'},
                {'b1': '../trainer/checkpoints/nn_3/epoch_0050/b1.npy'},
                {'w2': '../trainer/checkpoints/nn_3/epoch_0050/w2.npy'},
                {'b2': '../trainer/checkpoints/nn_3/epoch_0050/b2.npy'},
                {'w3': '../trainer/checkpoints/nn_3/epoch_0050/w3.npy'},
                {'b3': '../trainer/checkpoints/nn_3/epoch_0050/b3.npy'},
            ],
            'nn_4': [
                {'w1': '../trainer/checkpoints/nn_4/epoch_0050/w1.npy'},
                {'b1': '../trainer/checkpoints/nn_4/epoch_0050/b1.npy'},
                {'w2': '../trainer/checkpoints/nn_4/epoch_0050/w2.npy'},
                {'b2': '../trainer/checkpoints/nn_4/epoch_0050/b2.npy'},
                {'w3': '../trainer/checkpoints/nn_4/epoch_0050/w3.npy'},
                {'b3': '../trainer/checkpoints/nn_4/epoch_0050/b3.npy'},
            ],
            'nn_5': [
                {'w1': '../trainer/checkpoints/nn_5/epoch_0050/w1.npy'},
                {'b1': '../trainer/checkpoints/nn_5/epoch_0050/b1.npy'},
                {'w2': '../trainer/checkpoints/nn_5/epoch_0050/w2.npy'},
                {'b2': '../trainer/checkpoints/nn_5/epoch_0050/b2.npy'},
                {'w3': '../trainer/checkpoints/nn_5/epoch_0050/w3.npy'},
                {'b3': '../trainer/checkpoints/nn_5/epoch_0050/b3.npy'},
            ]
        }
        for _, params in model_weight_matrix_src.items():
            for param in params:
                w1 = param['w1']


if __name__ == '__main__':
    flt = StructureFilter()
    print(flt.samples.shape)
