import numpy as np
import matplotlib.pyplot as plt

# 定义原始曲线
ImpedanceA = 1
ImpedanceW = 1
x_values = np.linspace(0, 2*np.pi, 1000)
y_values = ImpedanceA * np.cos(ImpedanceW * x_values)

# 计算法线斜率
slope = 1/(ImpedanceW * ImpedanceA * np.sin(ImpedanceW * x_values))

# 计算法线点
width = 0.5
x_offset =np.sign(slope)* width / np.sqrt(1 + slope**2)
y_offset = slope * x_offset

# 计算放样后的点
x_left = x_values - x_offset
y_left = y_values - y_offset

# 绘制原始曲线
plt.plot(x_values, y_values, label='Original Curve')
plt.plot(x_left,y_left)

# 设置图形标题和坐标轴标签
plt.title('Extruded Curve')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
