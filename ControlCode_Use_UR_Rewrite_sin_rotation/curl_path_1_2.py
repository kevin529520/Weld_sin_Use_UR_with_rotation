'''
两条样条曲线且无偏差的解析方程
'''
from scipy.interpolate import interp1d
from scipy.interpolate import BSpline
import os
import matplotlib.pyplot as plt
import numpy as np

# from scipy.interpolate import CubicSpline

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
    angle_radians = np.radians(90)  # 89.69172888426066 #89.1062
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    x_path = np.zeros(0)
    y_path = np.zeros(0)
    x = np.zeros(50)
    y = np.linspace(-85, 25, 50)
    x_path = np.append(x_path, x)
    y_path = np.append(y_path, y)
    #
    ImpedanceW = 0.0314  # 31.4 #  # 频率
    ImpedanceA = 40  # 幅值
    PositionReal = np.zeros((200, 2))
    PositionReal[:, 1] = np.linspace(0, 2 * np.pi / ImpedanceW, 200)
    PositionReal[:, 0] = ImpedanceA * np.sin(ImpedanceW * PositionReal[:, 1])  # x方向
    x_path = np.append(x_path, PositionReal[:, 0])
    y_path = np.append(y_path, PositionReal[:, 1] + 25)
    y = np.linspace(225, 250 + 85 + 20, 50)
    x_path = np.append(x_path, x)
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

    # 按照 y_path 的值的大小进行排序
    sorted_indices = sorted(range(len(unique_y_path)), key=lambda k: unique_y_path[k], reverse=True)
    sorted_unique_y_path = [unique_y_path[i] for i in sorted_indices]
    sorted_unique_x_path = [unique_x_path[i] for i in sorted_indices]

    sorted_unique_x_path = np.array(sorted_unique_x_path)
    sorted_unique_y_path = np.array(sorted_unique_y_path)
    new_x_path = sorted_unique_x_path * cos_theta - sorted_unique_y_path * sin_theta
    new_y_path = sorted_unique_x_path * sin_theta + sorted_unique_y_path * cos_theta
    new_x_path = np.delete(new_x_path, slice(49, 50,))
    new_y_path = np.delete(new_y_path, slice(49, 50,))
    path_point = np.zeros(len(new_x_path) * 2)

    path_point[0::2] = new_x_path
    path_point[1::2] = new_y_path
    # PositionReal_with_random[:, 1] = ImpedanceA * np.sin(ImpedanceW * PositionReal_with_random[:, 0]) * np.random.uniform(
    #     0.98, 1.02, interpolation_number)

    cloud_points_spline = BSpline.interpolate_cubic_natural(path_point.tolist(), 2)
    # cloud_points_spline = CubicSpline(new_x_path, new_y_path, bc_type='natural')


    cloud_points_sample = cloud_points_spline.sample(9900)
    # x_samples = np.linspace(new_x_path.min(), new_x_path.max(), 9900)
    # y_samples = cloud_points_spline(x_samples)


    cloud_points = np.zeros((len(cloud_points_sample) // 2, 2))
    cloud_points[:, 0] = cloud_points_sample[0::2]
    cloud_points[:, 1] = cloud_points_sample[1::2]
    
    # cloud_points = np.zeros((len(x_samples), 2))
    # cloud_points[:, 0] = x_samples
    # cloud_points[:, 1] = y_samples


    gun_path_spline = cloud_points_spline
    gun_points = cloud_points.copy()
    ############################################

    sorted_unique_y_path = [unique_y_path[i] for i in sorted_indices]
    sorted_unique_x_path = [unique_x_path[i] + np.random.uniform(-1, 1) * 0.002 for i in sorted_indices]
    sorted_unique_x_path = np.array(sorted_unique_x_path)
    sorted_unique_y_path = np.array(sorted_unique_y_path)
    new_x_path = sorted_unique_x_path * cos_theta - sorted_unique_y_path * sin_theta
    new_y_path = sorted_unique_x_path * sin_theta + sorted_unique_y_path * cos_theta
    path_point = np.zeros((len(sorted_unique_x_path), 2))
    path_point[:, 0] = new_x_path
    path_point[:, 1] = new_y_path
    from Bézier_curve import bezier_curve
    curve_points = bezier_curve(path_point, seg=1000)
    finger_points_spline = []
    # finger_points_spline = cloud_points_spline
    finger_points = curve_points.copy()



    ###################333
    # fig4, ax4 = plt.subplots()
    # ax4.set_aspect('equal')
    # ax4.plot(finger_points[:, 0], finger_points[:, 1], 'bo',label='finger_points')
    # ax4.plot(gun_points[:, 0], gun_points[:, 1], label='gun_points')
    # ax4.plot([-x for x in unique_y_path], unique_x_path, label='origin_points')
    #
    # plt.legend()
    # plt.show()
    # plt.close()
    return gun_path_spline, gun_points, finger_points_spline, finger_points


if __name__ == '__main__':
    get_curl_path()
