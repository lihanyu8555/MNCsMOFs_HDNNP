# -*- coding = utf-8 -*-
# @Author：风城浪子
# @File：utils.py
# @Time：2023/2/16 10:20
# @Software：PyCharm

"""
    采用AENET神经网络的参数，截止半径选用6.5
"""


class DescriptorConfigure(object):
    """
            原子中心对称函数参数配置
        """
    # 径向函数选择组数
    radial_descriptor_numbers = 8
    # 角度函数选择组数
    angle_descriptor_numbers = 18
    # Au元素的截止半径
    AU_CUTOFF_RADIUS = 8
    # 径向函数参数设定
    radial_descriptor_params = {
        'eta': [0.006428, 0.012856, 0.025712, 0.051424, 0.102848, 0.205696, 0.411392, 0.822784],
        'rs': [0.0] * radial_descriptor_numbers,
        'rc': [AU_CUTOFF_RADIUS] * radial_descriptor_numbers,
    }
    # 角度函数参数设定
    angle_descriptor_params = {

        'eta': [0.000357, 0.005357, 0.010357, 0.015357, 0.020357, 0.025357, 0.030357, 0.035357, 0.040357, 0.045357,
                0.050357, 0.055357, 0.060357, 0.065357, 0.070357, 0.075357, 0.080357, 0.085357],
        'lambda_': [1, 1, 1, -1, -1, -1] * 3,
        'zeta': [2] * 6 + [3] * 6 + [4] * 6,
        'rc': [AU_CUTOFF_RADIUS] * angle_descriptor_numbers
    }
