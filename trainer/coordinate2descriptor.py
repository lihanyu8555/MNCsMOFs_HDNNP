# -*- coding = utf-8 -*-
# @Author：风城浪子
# @File：coordinate2descriptor.py
# @Time：2023/2/22 16:59
# @Software：PyCharm

"""
    坐标准换为描述符矩阵
"""
import os
import numpy as np
from descriptor.descriptor_func import AtomNeighborDescriptor


class CoordinateTransformer(object):
    xyz_src = [
        '../datasets/xyz_coordinate/Au13',
        '../datasets/xyz_coordinate/Au19',
        '../datasets/xyz_coordinate/Au38',
        '../datasets/xyz_coordinate/Au58',
        '../datasets/xyz_coordinate/Au75',
        '../datasets/xyz_coordinate/PdIr-MOF',
    ]
    descriptor_matrix_src = [
        '../datasets/descriptor_matrix/Au13_descriptor',
        '../datasets/descriptor_matrix/Au19_descriptor',
        '../datasets/descriptor_matrix/Au38_descriptor',
        '../datasets/descriptor_matrix/Au58_descriptor',
        '../datasets/descriptor_matrix/Au75_descriptor',
        '../datasets/descriptor_matrix/PdIr-MOF_descriptor',
                             ]
    def transform(self):
        """
            xyz三维坐标转换为描述符矩阵信息
        :return:
        """
        for fold_idx, src in enumerate(self.xyz_src):
            for idx, filename in enumerate(os.listdir(src)):
                filepath = src + '/' + filename
                with open(file=filepath, mode='r', encoding='utf-8') as fr:
                    lines = fr.readlines()
                # atoms = int(lines[0].strip())
                energy_value = float(lines[1].strip().split('=')[-1].strip('eV').strip())
                coordinate_matrix = []
                # 元素类型：C 0 H 1 N 2 O 3 Zr 4 Au 5 Pd 6 Ir 7
                atom_matrix = []
                for line in lines[2:]:
                    # atom_matrix.append(line.strip().split(' ')[:1])
                    atom = line.strip().split(' ')[:1]
                    if atom[0] == 'C':
                        atom_matrix.append(0)
                    elif atom[0] == 'H':
                        atom_matrix.append(1)
                    elif atom[0] == 'N':
                        atom_matrix.append(2)
                    elif atom[0] == 'O':
                        atom_matrix.append(3)
                    elif atom[0] == 'Zr':
                        atom_matrix.append(4)
                    elif atom[0] == 'Au':
                        atom_matrix.append(5)
                    elif atom[0] == 'Pd':
                        atom_matrix.append(6)
                    elif atom[0] == 'Ir':
                        atom_matrix.append(7)
                    else:
                        atom_matrix.append(8)
                    line = line.strip().split(' ')[1:]
                    coordinate_matrix.append([float(coordinate) for coordinate in line if coordinate.strip()])
                # atom_matrix = np.array(atom_matrix)
                atom_matrix = np.array(atom_matrix).reshape(-1, 1)
                # print(atom_matrix)
                coordinate_matrix = np.array(coordinate_matrix)
                coordinate_matrix = coordinate_matrix[:,0:3]
                energy_column = [energy_value] * coordinate_matrix.shape[0]
                energy_column = np.array(energy_column).reshape(-1, 1)
                # 计算径向函数矩阵
                radial_descriptor_matrix = dpt.retrieve_radial_descriptor_matrix(coordinate_matrix=coordinate_matrix)
                # print(radial_descriptor_matrix)
                # 计算角度函数矩阵
                angle_descriptor_matrix = dpt.retrieve_angle_descriptor_matrix(coordinate_matrix=coordinate_matrix)
                # print(angle_descriptor_matrix)
                descriptor_matrix = np.hstack((atom_matrix, radial_descriptor_matrix, angle_descriptor_matrix, energy_column))

                # print(descriptor_matrix)
                rename = filename.split('.')[0] + '.npy'
                # 数据存储
                with open(self.descriptor_matrix_src[fold_idx] + '/' + rename, mode='wb') as fp:
                    np.save(fp, descriptor_matrix)


if __name__ == '__main__':
    dpt = AtomNeighborDescriptor()
    transformer = CoordinateTransformer()
    transformer.transform()
