'''
两条样条曲线且无偏差的解析方程
'''
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import numpy as np
from tinyspline import *

np.random.seed(42587)


def rotate_points(x_path, y_path, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    new_x_path = x_path * cos_theta - y_path * sin_theta
    new_y_path = x_path * sin_theta + y_path * cos_theta
    return new_x_path, new_y_path


# Given data
def interpolate(x, y, t):
    m = interp1d(x, y)
    return m(t)


def get_curl_path():
    angle_radians = np.radians(0)  # 89.69172888426066 #89.1062
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    x_path = np.zeros(0)
    y_path = np.zeros(0)
    x = np.zeros(50)
    y = np.linspace(-85 - 44.5, 0, 50)
    x_path = np.append(x_path, x)
    y_path = np.append(y_path, y)
    #
    t = np.linspace(0, np.pi, 500)

    # 定义x和y作为t的函数
    x = 5 * np.sqrt(1 ** 2 * np.cos(2 * t) + np.sqrt(1 ** 4 * (np.cos(2 * t)) ** 2 + 0.5 ** 4)) * np.cos(t)
    y = 10 * np.sqrt(1 ** 2 * np.cos(2 * t) + np.sqrt(1 ** 4 * (np.cos(2 * t)) ** 2 + 0.5 ** 4)) * np.sin(t) - 1.8
    # 计算曲线的法向量
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    norm_x = -dy_dt / np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    norm_y = dx_dt / np.sqrt(dx_dt ** 2 + dy_dt ** 2)

    # 定义缩进距离
    indent_distance = 0.6

    # 沿法向方向缩进曲线
    x_indent = x + indent_distance * norm_x
    y_indent = y + indent_distance * norm_y

    # PositionReal[:, 1] = np.linspace(0.0001, 2 * np.pi / ImpedanceW, 200)
    # PositionReal[:, 0] = ImpedanceA * np.cos(ImpedanceW * PositionReal[:, 1])  # x方向
    x_left = x_indent - 6.525272045737943
    y_left = y_indent + 0.027767716082087723
    x_left = x_left * 10
    y_left = y_left * 10
    x_path = np.append(x_path, x_left)
    y_path = np.append(y_path, y_left)
    y = np.linspace(0, -85 - 44.5, 50)

    x_path = np.append(x_path, np.ones_like(y) *-130.50544091475888)
    y_path = np.append(y_path, y)
    unique_values = {}
    for x, y in zip(x_path, y_path):
        if y not in unique_values:
            unique_values[y] = x

    # 提取去重后的 x_path 和 y_path
    unique_y_path = list(unique_values.keys())
    unique_x_path = list(unique_values.values())
    unique_y_path = [x / 1000 for x in unique_y_path]
    unique_x_path = [x / 1000 for x in unique_x_path]
    unique_x_path = np.array(unique_x_path)
    unique_y_path = np.array(unique_y_path)
    new_x_path = unique_x_path * cos_theta - unique_y_path * sin_theta
    new_y_path = unique_x_path * sin_theta + unique_y_path * cos_theta
    path_point = np.array([new_x_path, new_y_path]).T
    from Bézier_curve import bezier_curve
    curve_points = bezier_curve(path_point, seg=1000)
    finger_points_spline = []
    # finger_points_spline = cloud_points_spline
    finger_points = curve_points.copy()
    fig4, ax4 = plt.subplots()
    ax4.set_aspect('equal')
    ax4.plot(finger_points[:, 0], finger_points[:, 1], 'bo', label='finger_points')

    # ax4.plot([-x for x in unique_y_path], unique_x_path, label='origin_points')

    plt.legend()
    plt.show()
    # plt.close()
    return None, None, finger_points_spline, finger_points


if __name__ == '__main__':
    get_curl_path()
