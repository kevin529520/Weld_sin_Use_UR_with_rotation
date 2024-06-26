import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def compute_bezier_curve(control_points, t):
    n = len(control_points) - 1
    x = 0
    y = 0
    for i in range(n + 1):
        coefficient = comb(n, i) * (1 - t) ** (n - i) * t ** i
        x += coefficient * control_points[i][0]
        y += coefficient * control_points[i][1]
    return x, y


def bezier_curve(control_points, seg=200):
    t = np.linspace(0, 1, seg)
    curve_points = np.array([compute_bezier_curve(control_points, ti) for ti in t])
    return curve_points


if __name__ == '__main__':
    # 定义控制点
    control_points = np.array([[0, 0], [10, 2], [3, 1], [4, 3]])

    # 绘制控制点
    plt.scatter(control_points[:, 0], control_points[:, 1], c='r', label='Control Points')

    # 计算并绘制贝塞尔曲线
    curve_points = bezier_curve(control_points)
    plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bézier Curve')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bézier Curve Passing Through Endpoints')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()
