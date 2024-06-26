import math
import multiprocessing
import threading
import time

import multiprocessing as mp
import numpy as np
# from ControlCode.ReadOnrobot import OnRobotFT
from scipy import signal
import os
import queue
from URClient import RobotClient
from scipy.spatial.transform import Rotation as Rt
from curl_path_1_2 import get_curl_path
import matplotlib.pyplot as plt
from Bézier_curve import bezier_curve


def cal_nearestPoint_bezier(point_array, point):
    selected_points = point_array[point_array[:, 0] < point[0]]
    # 计算选取的点离 pose 最近的点
    distances = np.linalg.norm(selected_points - point, axis=1)
    # nearest_point_indices = np.argsort(distances)[:2]
    nearest_point_index = np.argmin(distances)
    # print(nearest_point_index)
    # print(point_array[nearest_point_index - 1, :])
    x1, y1 = point_array[nearest_point_index, :]
    x2, y2 = point_array[nearest_point_index + 3, :]
    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)
    tangent_direction = np.arctan(slope)
    # print("tangent_direction =", tangent_direction)

    tangential_x = -np.cos(tangent_direction)
    tangential_y = -np.sin(tangent_direction)

    normal_direction = tangent_direction + np.pi / 2  # 与切线方向垂直
    normal_x = np.cos(normal_direction)
    normal_y = np.sin(normal_direction)
    # if y1 > 0:
    #     normal_x = -np.cos(normal_direction)
    #     normal_y = -np.sin(normal_direction)
    # else:
    #     normal_x = np.cos(normal_direction)
    #     normal_y = np.sin(normal_direction)
    # 计算法向方向的x、y分量

    if selected_points.__len__() < 2:
        return False
    return point_array[nearest_point_index - 3, :], normal_x, normal_y, tangential_x, tangential_y


def find_nearest_points_2(target_point, points_array):
    # 计算目标点与所有点之间的距离
    distances = np.linalg.norm(points_array[:-2, :] - target_point, axis=1)
    nearest_point_index = np.argmin(distances)
    x1, y1 = points_array[nearest_point_index + 2, :]
    x2, y2 = points_array[nearest_point_index, :]
    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)
    tangent_direction = np.arctan(slope)

    tangential_x = -np.cos(tangent_direction)
    tangential_y = -np.sin(tangent_direction)

    normal_direction = tangent_direction + np.pi / 2  # 与切线方向垂直
    normal_x = np.cos(normal_direction)
    normal_y = np.sin(normal_direction)

    return points_array[nearest_point_index, :], normal_x, normal_y, tangential_x, tangential_y


def find_nearest_points(target_point, points_array, k=5):
    # 计算目标点与所有点之间的距离
    distances = np.linalg.norm(points_array - target_point, axis=1)

    # 按距离升序排序，并返回前k个最近的点的索引
    nearest_indices = np.argsort(distances)[:k]
    nearest_indices.sort()
    # 返回最近的点的索引和距离
    nearest_points = points_array[nearest_indices]
    x1, y1 = nearest_points[0, :]
    x2, y2 = nearest_points[-1, :]
    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)
    tangent_direction = np.arctan(slope)

    tangential_x = -np.cos(tangent_direction)
    tangential_y = -np.sin(tangent_direction)

    normal_direction = tangent_direction + np.pi / 2  # 与切线方向垂直
    normal_x = np.cos(normal_direction)
    normal_y = np.sin(normal_direction)

    return normal_x, normal_y, tangential_x, tangential_y


def rpy2rot_vec(rpy):
    r = Rt.from_euler('xyz', rpy)
    rot_vec = r.as_rotvec()
    return rot_vec
    # 欧拉角转换成旋转向量


def dealForceCustomImpedance(Fxe, Fs, dvk, vk, x, timeLast, m, b, k):
    FPlusDF = Fxe + Fs
    currentTime = time.time()
    dt = currentTime - timeLast
    # print(dt)
    if dt > 0.5:
        print("RockControlUsingFinger dt =", dt)
        dt = 0.5
    dvkPlus = ((FPlusDF - b * vk - k * x) / m) # 加速度
    vkPlus = dt * (dvk + dvkPlus) / 2 + vk
    return dvkPlus, vkPlus, currentTime, dt
# 自定义阻抗控制中的力反馈。这是一个经典的阻抗控制模型，通常用于机器人控制等领域。阻抗控制通过调节虚拟质量、阻尼和刚度参数，使得系统对外部力的响应类似于一个弹簧-阻尼-质量系统。


def AdmittanceControl(dataDir, InitTime, gun_path_spline, gun_points, finger_points_spline, finger_points,
                      FingerForceReceiveQue=mp.Queue,
                      stopQue=mp.Queue,
                      ImpedanceControl=False, ImpedanceControl3=False, ):
    
    # ForceReceive = FingerForceReceiveQue.get(timeout=10)


    # FingerForceReceiveQue.put(np.concatenate((OffsetForce, ArucoPose), axis=0))
    # 通过将其定义为一个多进程队列，可以确保数据的安全传递和同步访问，避免多个进程同时访问数据时出现竞争条件


    Force_init = np.zeros(1)

    FingerForceRecord = np.zeros((0, 6))
    SGVelRecord = np.zeros((0, 6))
    PositionDesire = np.zeros(6)
    dt = np.zeros(6)
    vk = np.zeros(6)
    dvk = np.zeros(6)
    Fs_n = np.zeros(2)
    Fs_t = np.zeros(2)
    Fs = np.zeros(2)
    timeLast = np.zeros(6)
    ########### ControlProcess ########
    robot = RobotClient()
    tcp = rpy2rot_vec((-np.pi, 0, 0))
    # [-3.14159265  0.          0.        ]

    now_pose = robot.ReadMachinePosition()[0:3]
    now_pose[2] = now_pose[2] + 0.1

    robot.Move(now_pose[0:3], tcp)

    
    # 移动到哪个位置
    # tcp 移动时的转角

    # -0.016139876756485903, -0.8102090933335253, 0.3722249608780756
    # Q_joint = [1.6538190841674805,
    #            -0.9526007932475586,
    #            1.6189902464496058,
    #            4.046006842250488,
    #            4.710605621337891,
    #            2.4786343574523926]
    Q_joint = [1.6454973220825195, -1.6632138691344203, 2.2385829130755823, -2.71738400081777, 4.661285877227783, 3.1862950325012207]
    
    # Q_joint = [1.49882173538208, -1.2290848058513184, 2.156137768422262, -2.5010653934874476, 4.708913803100586, 0.02007436752319336]
    # 方法将这些目标位置或角度应用于机器人的相应关节，从而控制机器人的整体姿态和位置。
    # 1. 基座旋转（Base rotation） - 控制机器人基座的旋转。
    # 2. 肩部倾斜（Shoulder tilt） - 控制机器人的“肩部”部分的倾斜。
    # 3. 肘部旋转（Elbow rotation） - 控制机器人的“肘部”部分的旋转。
    # 4. 腕部旋转（Wrist rotation） - 控制机器人的“腕部”部分的旋转。
    # 5. 腕部倾斜（Wrist tilt） - 控制机器人腕部的倾斜。
    # 6. 腕部偏转（Wrist swivel） - 控制机器人腕部的偏转。

    robot.moveJoint(Q_joint)
    init_pose = robot.ReadMachinePosition()[0:3]
    init_pose[0] = init_pose[0]
    init_pose[1] = init_pose[1]

    robot.Move(init_pose, tcp)
    # # 目标位置以及角度 是否会有动作还取决于 tcp 参数，即目标姿态或工具中心点方向。如果 tcp 指定的方向与机器人当前的末端执行器方向不同，那么机器人可能会进行旋转或调整姿态以匹配新的方向指示，即使它的位置坐标没有变化。


    # init_pose[2] = init_pose[2] - 0.1 - 0.003
    # # #
    # robot.Move(init_pose, tcp)
    # init_pose[1] = init_pose[1] - 0.0045

    # robot.Move(init_pose, tcp)

    time.sleep(1)
    ForceReceive = FingerForceReceiveQue.get(timeout=0.1)
    # mp.Queue 队列是多进程中的队列，用于在不同的进程或线程之间传递数据。
    # 在这种情况下，它用于从 FingerForceReceiveQue 队列中获取数据，如果队列为空，则会阻塞等待数据。
    # 如果队列中没有数据，则会抛出 queue.Empty 异常。
    print("ForceReceive in RockControl", ForceReceive)
    offset_y = abs(ForceReceive[7]) * 3.5 
    # 指尖偏移 = 二维码y方向的偏移量 * 3.5（dp[1]） * ((22.37+56.24)/22.37=3.51)

    tcp = rpy2rot_vec((-np.pi, 0, np.arctan(offset_y / 0.085)))
    robot.Move(init_pose, tcp)
    time.sleep(1)


    plt.rcParams['lines.markersize'] = 3
    RecordControlTime = np.zeros(0)

    x1 = np.linspace(0, 0.09, 10)[::-1]
    # [::-1] 是一个切片操作符，具体解释如下：
    # : 表示从头到尾的切片。
    # # : 表示默认的步长（即 1）。
    # # -1 表示步长为 -1，即从后向前逐步取值。

    y = np.ones_like(x1) * (offset_y)
    Finger_tail_end_array = np.array([x1, y]).T
    # 将 x1 和 y 的值转换为一个二维数组，其中 x1 是行，y 是列。

    FingerTailMachinePosition = np.zeros((0, 3))
    # array([], shape=(0, 3), dtype=float64)
    FingerTailArucoPose = np.zeros((0, 3))
    TcpForceArray = np.zeros((0, 6))
    Count = 0

    pose = robot.ReadMachinePosition()
    # 读取机器人的当前位置

    InitPosition = pose[0:3]
    MachinePosition = robot.ReadMachinePosition()[0:3] - InitPosition
    
    print('MachinePosition', MachinePosition)
    BTime = time.time()
    path_begin = True
    theta_array=[]
    omega_array = []
    for m in range(6):
        timeLast[m] = time.time()
    first_loop = True

    print("RockControlUsingFinger start")

    while stopQue.qsize() is 0:
        #  队列的大小为 0 
        try:
            ForceReceive = FingerForceReceiveQue.get(timeout=0.1)
            # 如果队列为空，它会等待一段时间（这里是 0.1 秒）直到有数据可用。
            # print("FingerForceReceiveQue:",FingerForceReceiveQue.qsize())
        except queue.Empty:
            print("FingerForceReceiveQue timeout!")
            continue
        FingerForce = ForceReceive[0:6]
        # FingerForce[2] = 0
        pose = robot.ReadMachinePosition()
        r = Rt.from_rotvec(pose[3:6])
        rpy = r.as_euler('xyz', degrees=False)
        theta = rpy[-1]
        # 机器人末端的欧拉角

        ArucoPose = ForceReceive[6:]
        # 相机坐标系下的二维码姿态
        # ForceReceive FingerForceReceiveQue获得力 and 二维码姿态
        MachinePosition = pose[0:3]
        Position = MachinePosition - InitPosition
        # position为相对初始position的变化
        print(ArucoPose[0:2])
        ArucoPose[1] = -ArucoPose[1]
        # 为何换成相反数？
        ArucoPose[0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]) @ ArucoPose[0:2]
        # 相机坐标系换到机器人坐标系
        Finger_tail_end = Position[0:2] + ArucoPose[0:2] * 3.5
        # 为何乘以3.5
        FingerForce[1] = -FingerForce[1]
        FingerForce[0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]]) @ FingerForce[0:2]
        # 对两个向量 ArucoPose 和 FingerForce 进行变换和旋转操作：
        # 1. 对 ArucoPose 进行变换和旋转操作，使其在机器人的坐标系下表示。(参考)
        # 2. 对 FingerForce 进行变换和旋转操作，使其在机器人的坐标系下表示。（参考）

        # print(FingerForce)
        # FingerForce = np.zeros(6)
        if ImpedanceControl3 is True:
            b = 0.002
            finger_pose = Position
            if abs(finger_pose[0]) > 0.248:
                b = 0
            return_ = cal_nearestPoint_bezier(finger_points, finger_pose[0:2])
            if not return_:
                stopQue.put('1') 
                continue
            PositionDesire, normal_x, normal_y, tangential_x, tangential_y = return_
            FingerForce_normal_projection = FingerForce[0] * normal_x + FingerForce[1] * normal_y
            # fingerforce就是ypredict，且转换到了机器人坐标系下
            if first_loop is True:
                first_loop = False
                Force_init = FingerForce_normal_projection
                # 初始形变产生的力
            Force_normal_projection_refer = Force_init
            v_normal = b * (FingerForce_normal_projection - Force_normal_projection_refer)
            v_tangent = 0.005
            vk[0] = v_normal * normal_x + v_tangent * tangential_x
            vk[1] = v_normal * normal_y + v_tangent * tangential_y

            # Cal omega
            L = 0.085
            r_x = L * np.cos(theta)
            r_y = L * np.sin(theta)
            Gun_pose = np.array([finger_pose[0] + r_x, finger_pose[1] + r_y])
            if abs(FingerForce_normal_projection - Force_normal_projection_refer) > 0.1 and path_begin is True:
                omega = 0
                print("abs(FingerForce_normal_projection - Force_normal_projection_refer) > 0.1 ")
            else:
                path_begin = False
                normal_gun_x, normal_gun_y, tangential_gun_x, tangential_gun_y = find_nearest_points(Gun_pose,
                                                                                                     Finger_tail_end_array)
                # print(normal_gun_x, normal_gun_y, tangential_gun_x, tangential_gun_y)
                v_gun_normal_1 = vk[0] * normal_gun_x + vk[1] * normal_gun_y
                omega = -v_gun_normal_1 / (L * -np.sin(theta) * normal_gun_x + L * np.cos(theta) * normal_gun_y)
                # (L * (-np.sin(theta) * tangential_gun_x + np.cos(theta) * tangential_gun_y))
            vk[-1] = omega
            # print(omega)
        # vk[-1] = 0
        SGVel = vk.copy()
        # SGVel[1] = -SGVel[1]
        gap = 0.0001
        SGVel[:2] = np.where(np.abs(SGVel[:2]) < gap, 0, np.sign(SGVel[:2]) * (np.abs(SGVel[:2])))
        # SGVel[:2] = np.where(np.abs(SGVel[:2]) < gap, 0, SGVel[:2])

        SGVel[-1] = np.where(np.abs(SGVel[-1]) < gap * 10, 0, np.sign(SGVel[-1]) * (np.abs(SGVel[-1])))
        # 使用 np.where 函数对 SGVel 的前两个分量（通常是 x 和 y 方向的线速度）进行处理：
        # np.abs(SGVel[:2]) < gap：检查速度分量的绝对值是否小于阈值 gap。
        # 如果条件为真（即速度分量接近于零），则将该分量设置为 0。
        # 否则，保持原始值不变。使用 np.sign(SGVel[:2]) * (np.abs(SGVel[:2])) 确保速度的符号和大小不变。


        # print(SGVel[0], SGVel[1], SGVel[2], SGVel[3], SGVel[4], SGVel[5])
        robot.Speed(SGVel[0], SGVel[1], SGVel[2], SGVel[3], SGVel[4], SGVel[5], )
        # plt.plot(Count,omega,'ro')
        plt.plot(Position[0], Position[1], 'ro')  # 'ro'表示红色圆点
        plt.plot(Finger_tail_end[0], Finger_tail_end[1], 'bo')
        plt.plot(Gun_pose[0], Gun_pose[1], 'yo')  # 'ro'表示红色圆点

        #
        # plt.quiver(PositionDesire[0], PositionDesire[1], normal_x, normal_y, scale=20, color='blue',
        #            label='Normal Direction')
        # plt.quiver(PositionDesire[0], PositionDesire[1], tangential_x, tangential_y, scale=20, color='g',
        #            label='Tangent Direction')
        plt.axis('equal')

        # 更新图形
        plt.draw()
        plt.pause(0.01)  # 暂停0.1秒，以便看到更新的图形
        SGVelRecord = np.concatenate((SGVelRecord, SGVel.reshape(1, 6)), axis=0)
        FingerForceRecord = np.concatenate((FingerForceRecord, FingerForce.reshape(1, 6)), axis=0)

        RecordControlTime = np.append(RecordControlTime, time.time() - InitTime)
        FingerTailMachinePosition = np.concatenate(
            (FingerTailMachinePosition, MachinePosition.reshape(1, 3)), axis=0)
        FingerTailArucoPose = np.concatenate(
            (FingerTailArucoPose, ArucoPose[0:3].reshape(1, 3)), axis=0)
        if abs(FingerForce_normal_projection - Force_normal_projection_refer) > 0.1 and path_begin is True:
            print("abs(FingerForce_normal_projection - Force_normal_projection_refer) > 0.1 2")
            pass
        else:
            Finger_tail_end_array = np.concatenate(
                (Finger_tail_end_array, Finger_tail_end[0:2].reshape(1, 2)), axis=0)
        theta_array.append(theta)
        omega_array.append(omega)
        Count = Count + 1

    robot.close()
    time.sleep(1)
    np.save(dataDir + 'FingerTailMachinePosition.npy', FingerTailMachinePosition, )
    np.save(dataDir + 'FingerTailArucoPose.npy', FingerTailArucoPose, )
    np.save(dataDir + 'RecordControlTime.npy', RecordControlTime, )
    np.save(dataDir + 'TcpForceArray.npy', TcpForceArray, )
    np.save(dataDir + 'thetaArray.npy', np.array(theta_array), )
    np.save(dataDir + 'omegaArray.npy', np.array(omega_array), )
    np.save(dataDir + 'Force_y_init.npy', np.array(Force_init), )

    np.save(dataDir + 'SGVel.npy', SGVelRecord)
    np.save(dataDir + 'FingerForce.npy', FingerForceRecord)
