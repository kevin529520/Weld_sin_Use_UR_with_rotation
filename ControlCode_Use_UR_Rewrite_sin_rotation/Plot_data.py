import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from curl_path_1_2 import get_curl_path
import scienceplots
import os
plt.rcParams['lines.linewidth'] = 5  # Change the value to adjust the linewidth as needed
plt.rcParams['font.size'] = 15
# plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体

gun_path_spline, gun_points, finger_points_spline, finger_points = get_curl_path()
# 定义数据目录
# data_dir = "../../data/焊道1_test_for_plot/"
data_dir = "../../data/0320222020/"
output_dir = os.path.join(data_dir, "saved_data")
os.makedirs(output_dir, exist_ok=True)
# 读取.npy文件
finger_force = np.load(data_dir + "FingerForce.npy")
aruco_pose = np.load(data_dir + "ArucoPose.npy")
record_offset_force_time = np.load(data_dir + "RecordOffsetForceTime.npy")
record_control_time = np.load(data_dir + "RecordControlTime.npy")
machine_position = np.load(data_dir + "FingerTailMachinePosition.npy")
FingerTailArucoPose = np.load(data_dir + "FingerTailArucoPose.npy")
OffsetForce = np.load(data_dir+"OffsetForce.npy")
# 打印数组的形状（可选）
print("FingerForce shape:", finger_force.shape)
print("ArucoPose shape:", aruco_pose.shape)
print("RecordOffsetForceTime shape:", record_offset_force_time.shape)
print("RecordControlTime shape:", record_control_time.shape)
fig_width = 12
fig_height = 6
plt.figure(figsize=(fig_width, fig_height))
# 计算FingerForce合力
total_force = np.linalg.norm(OffsetForce[:, :2], axis=1)

# 创建DataFrame
df = pd.DataFrame({'RecordOffsetForceTime': record_offset_force_time, 'Total_Force': total_force})
np.save(os.path.join(output_dir, "force_time.npy"), record_offset_force_time)
np.save(os.path.join(output_dir, "total_force.npy"), total_force)
np.save(os.path.join(output_dir, "force.npy"), OffsetForce[:, :2])

# 仅保留0到75秒的数据
df_filtered = df[(df['RecordOffsetForceTime'] >= 0) & (df['RecordOffsetForceTime'] <= 750)]
plt.plot(df_filtered['RecordOffsetForceTime'],df_filtered['Total_Force'])
# 绘制线图
# sns.lineplot(data=df_filtered, x='RecordControlTime', y='Total_Force')


# 设置图形标题和坐标轴标签
plt.title('Time-Total_Force')
plt.xlabel('Time')
plt.ylabel('Total_Force')
plt.savefig(data_dir+"force.png")

# 显示图形
plt.show()
plt.rcParams['lines.linewidth'] = 2  # Change the value to adjust the linewidth as needed

fig_width = 12
fig_height = 6
plt.figure(figsize=(fig_width, fig_height))
# Extract (x, y) coordinates
x_machine = machine_position[:240, 0] - machine_position[0, 0]
y_machine = machine_position[:240, 1] - machine_position[0, 1]-0.006

np.save(os.path.join(output_dir, "Robot_end_position.npy"), np.array([x_machine,y_machine]))
x_finger = x_machine + FingerTailArucoPose[:240, 0] * 0.3
y_finger = y_machine + FingerTailArucoPose[:240, 1] * 0.3
plt.plot(x_finger, y_finger, label='Finger Tail-End Trajectory')
np.save(os.path.join(output_dir, "Finger_end_position.npy"), np.array([x_finger,y_finger]))
np.save(os.path.join(output_dir, "Robot_control_time.npy"), record_control_time)

# 定义原始曲线
x_real = finger_points[:, 0]
y_real = finger_points[:, 1]

# 定义 x_real 区间
a, b = -0.25, 0  # 选择适当的 a 和 b 值

# 根据 x_real 的区间进行筛选
mask = (x_real > a) & (x_real < b)
x_selected = x_real[mask]
y_selected = y_real[mask]
np.save(os.path.join(output_dir, "Obstacle_position.npy"), np.array([x_selected,y_selected]))

# Plot (x, y) coordinates
plt.plot(x_machine, y_machine, label='Robot End-Effector Trajectory')

# Plot finger points
plt.plot(x_selected, y_selected, label='Obstacle Trajectory')
# Set plot title and axis labels
plt.title('Robot End-Effector & Finger Tail-End Position')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
# Show plot
plt.grid(True)
plt.savefig(data_dir+"pose.png")

plt.show()
