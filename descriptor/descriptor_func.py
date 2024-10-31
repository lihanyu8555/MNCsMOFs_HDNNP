# -*- coding = utf-8 -*-
# @Author：风城浪子
# @File：descriptor_func.py
# @Time：2023/2/16 10:24
# @Software：PyCharm

"""
    原子坐标转换为描述符矩阵
"""
import os
import numpy as np
from descriptor.utils import DescriptorConfigure


class AtomNeighborDescriptor(object):
    @staticmethod
    def cutoff_radius_function(distance_ij, rc):
        """
            截止函数表示
        :param distance_ij: 原子i和原子j之间的距离
        :param rc: 截止半径
        :return:
        """
        return 0.5 * (1 + np.cos(np.pi * distance_ij / rc)) if distance_ij <= rc else 0

    def radial_function_descriptor(self, coordinate_matrix, eta, rs, rc):
        """
                    指定一组径向函数参数计算
                :param coordinate_matrix: 坐标矩阵
                :param eta: 径向参数
                :param rs: 径向参数
                :param rc: 截止半径
                :return: 径向函数值
                """
        descriptor_vec = []
        # 迭代每个原子
        for i in range(int(coordinate_matrix.shape[0])):
            temp = 0
            # 计算截止半径内的所有原子的径向函数值
            for j in range(coordinate_matrix.shape[0]):
                if i == j:
                    continue
                # 计算原子间距离
                distance = np.linalg.norm(coordinate_matrix[i] - coordinate_matrix[j])
                # 径向函数描述符计算公式
                temp += np.exp(-eta * np.power(distance - rs, 2) / np.power(rc, 2)) * self.cutoff_radius_function(
                    distance, rc=rc)
            descriptor_vec.append(temp)
        return np.array(descriptor_vec).T

    def retrieve_radial_descriptor_matrix(self, coordinate_matrix):
        """
            计算给定组数的径向函数描述符矩阵
        :param coordinate_matrix: 坐标矩阵  atom_matrix:元素类型标签
        :return: 描述符矩阵
        """
        radial_descriptor_matrix = []
        # 给定8组径向函数参数
        for instance in list(
                zip(DescriptorConfigure.radial_descriptor_params['eta'],
                    DescriptorConfigure.radial_descriptor_params['rs'],
                    DescriptorConfigure.radial_descriptor_params['rc'])):
            params = dict()
            params['eta'] = instance[0]
            params['rs'] = instance[1]
            params['rc'] = instance[2]
            # 计算单一组数径向描述符值
            radial_descriptor_vec = self.radial_function_descriptor(coordinate_matrix, **params)
            # 8组径向描述符函数值合并为numpy矩阵
            radial_descriptor_matrix.append(radial_descriptor_vec)
        return np.array(radial_descriptor_matrix).T

    @staticmethod
    def angle_cos_value(vector_1, vector_2):
        """
            计算两个向量之间的夹角的余弦值
        :return:
        """
        return np.dot(vector_1, vector_2) / (
                np.linalg.norm(vector_1, ord=2) * np.linalg.norm(vector_2, ord=2))

    def angle_descriptor(self, coordinate_matrix, zeta, lambda_, eta, rc):
        """
                    高斯角度函数计算
                :param coordinate_matrix: 坐标矩阵
                :param zeta: 晶格参数
                :param lambda_: 晶格参数
                :param eta: 晶格参数
                :param rc: 截止半径
                :return:
        """
        descriptor_vec: list = []
        for i in range(coordinate_matrix.shape[0]):
            temp = 0
            for j in range(coordinate_matrix.shape[0]):
                if i != j:
                    for k in range(coordinate_matrix.shape[0]):
                        if k != i and k != j:
                            # 计算各原子之间的欧式距离
                            distance_ij = np.linalg.norm(coordinate_matrix[i] - coordinate_matrix[j])
                            distance_ik = np.linalg.norm(coordinate_matrix[i] - coordinate_matrix[k])
                            distance_jk = np.linalg.norm(coordinate_matrix[j] - coordinate_matrix[k])
                            # 计算以i原子为中心，ij和ik向量之间的夹角
                            r_ij_vector = coordinate_matrix[j] - coordinate_matrix[i]
                            r_ik_vector = coordinate_matrix[k] - coordinate_matrix[i]
                            # 计算两个向量余弦值夹角
                            cos_angle_ijk = self.angle_cos_value(r_ij_vector, r_ik_vector)
                            # G3描述符计算公式
                            temp += np.power(1 + lambda_ * cos_angle_ijk, zeta) * np.exp(
                                -eta * (distance_ij ** 2 + distance_ik ** 2 + distance_jk ** 2)) * \
                                    self.cutoff_radius_function(distance_ij, rc=rc) * self.cutoff_radius_function(
                                distance_ik, rc=rc) * self.cutoff_radius_function(distance_jk, rc=rc)
            descriptor_vec.append(2 ** (1 - zeta) * temp)
        return descriptor_vec

    def retrieve_angle_descriptor_matrix(self, coordinate_matrix):
        # 计算多组角度函数
        angle_descriptor_matrix = []
        # 参数选择
        for instance in list(
                zip(DescriptorConfigure.angle_descriptor_params['eta'],
                    DescriptorConfigure.angle_descriptor_params['zeta'],
                    DescriptorConfigure.angle_descriptor_params['lambda_'],
                    DescriptorConfigure.angle_descriptor_params['rc'])):
            params = dict()
            params['eta'] = instance[0]
            params['zeta'] = instance[1]
            params['lambda_'] = instance[2]
            params['rc'] = instance[3]
            # 角度函数进行计算
            angle_descriptor_vec = self.angle_descriptor(coordinate_matrix, **params)
            # 返回角度函数特征向量矩阵
            angle_descriptor_matrix.append(angle_descriptor_vec)
        return np.array(angle_descriptor_matrix).T

# if __name__ == '__main__':
#     """描述符转换"""
#     dpt = AtomNeighborDescriptor()
#     sample_path = ['../dataset/npy/Au13_opt', '../dataset/npy/Au19_opt', '../dataset/npy/Au38_opt']
#     save_path = ['../descriptor/matrix/Au13_opt', '../descriptor/matrix/Au19_opt', '../descriptor/matrix/Au38_opt']
#     for fold_idx, src in enumerate(sample_path):
#         for idx, filename in enumerate(os.listdir(src)):
#             filepath = src + '/' + filename
#             print(filepath)
#             with open(filepath, mode='rb') as fr:
#                 dataset = np.load(fr)
#             # 取出坐标矩阵-转换为描述符矩阵
#             coordinate_matrix = dataset[:, :3]
#             # 计算径向函数矩阵和角度函数矩阵
#             radial_descriptor_matrix = dpt.retrieve_radial_descriptor_matrix(coordinate_matrix)
#             angle_descriptor_matrix = dpt.retrieve_angle_descriptor_matrix(coordinate_matrix=coordinate_matrix)
#             print(np.hstack((radial_descriptor_matrix, angle_descriptor_matrix)))
#             # angle_descriptor_matrix = dpt.retrieve_angle_descriptor_matrix(coordinate_matrix)
#             # descriptor_matrix = np.hstack(
#             #     (radial_descriptor_matrix, angle_descriptor_matrix, dataset[:, -1].reshape(-1, 1)))
#             # rename = filename.split('.')[0] + '.npy'
#             # # 数据存储
#             # with open(save_path[fold_idx] + '/' + rename, mode='wb') as fp:
#             #     np.save(fp, descriptor_matrix)
#             break
#         break
