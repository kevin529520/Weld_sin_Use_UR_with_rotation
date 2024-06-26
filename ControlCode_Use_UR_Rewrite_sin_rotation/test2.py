import numpy as np
import matplotlib.pyplot as plt

# 定义 t 的范围
t_values = np.linspace(0, 2*np.pi, 100)  # 使用100个点进行绘制

# 计算每个点的 x 和 y 坐标
x_values = np.cos(t_values)
y_values = np.sin(t_values)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='轨迹')
plt.scatter(x_values, y_values, color='red', label='点', s=10)  # 绘制点
plt.xlabel('x')
plt.ylabel('y')
plt.title('x 和 y 坐标随 t 的变化')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 使 x 和 y 轴比例相等
plt.show()
