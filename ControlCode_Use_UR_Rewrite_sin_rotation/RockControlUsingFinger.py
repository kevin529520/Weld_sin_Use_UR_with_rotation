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


def dealForceCustomImpedance(Fxe, Fs, dvk, vk, x, timeLast, m, b, k):
    FPlusDF = Fxe + Fs
    currentTime = time.time()
    dt = currentTime - timeLast
    # print(dt)
    if dt > 0.5:
        print("RockControlUsingFinger dt =", dt)
        dt = 0.5
    dvkPlus = ((FPlusDF - b * vk - k * x) / m)
    vkPlus = dt * (dvk + dvkPlus) / 2 + vk
    return dvkPlus, vkPlus, currentTime, dt


def AdmittanceControl(dataDir, InitTime, gun_path_spline, gun_points, finger_points_spline, finger_points,
                      FingerForceReceiveQue=mp.Queue,
                      stopQue=mp.Queue,
                      ImpedanceControl=False, ImpedanceControl3=False, ):
    ForceReceive = FingerForceReceiveQue.get(timeout=10)

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
    now_pose = robot.ReadMachinePosition()[0:3]
    now_pose[2] = now_pose[2] + 0.1

    robot.Move(now_pose[0:3], tcp)

    # -0.016139876756485903, -0.8102090933335253, 0.3722249608780756
    Q_joint = [1.6538190841674805,
               -0.9526007932475586,
               1.6189902464496058,
               4.046006842250488,
               4.710605621337891,
               2.4786343574523926]
    robot.moveJoint(Q_joint)
    init_pose = robot.ReadMachinePosition()[0:3]
    init_pose[0] = init_pose[0]
    init_pose[1] = init_pose[1]

    robot.Move(init_pose, tcp)
    init_pose[2] = init_pose[2] - 0.1 - 0.003
    # #
    robot.Move(init_pose, tcp)
    init_pose[1] = init_pose[1] - 0.0045

    robot.Move(init_pose, tcp)


    time.sleep(1)
    ForceReceive = FingerForceReceiveQue.get(timeout=0.1)
    print(ForceReceive)
    offset_y = abs(ForceReceive[7]) * 3.5

    tcp = rpy2rot_vec((-np.pi, 0, np.arctan(offset_y / 0.085)))
    robot.Move(init_pose, tcp)
    time.sleep(1)


    plt.rcParams['lines.markersize'] = 3
    RecordControlTime = np.zeros(0)

    x1 = np.linspace(0, 0.09, 10)[::-1]

    y = np.ones_like(x1) * (offset_y)
    Finger_tail_end_array = np.array([x1, y]).T
    FingerTailMachinePosition = np.zeros((0, 3))
    FingerTailArucoPose = np.zeros((0, 3))
    TcpForceArray = np.zeros((0, 6))
    Count = 0

    pose = robot.ReadMachinePosition()

    InitPosition = pose[0:3]
    MachinePosition = robot.ReadMachinePosition()[0:3] - InitPosition
    print(MachinePosition)
    BTime = time.time()
    path_begin = True
    theta_array=[]
    omega_array = []
    for m in range(6):
        timeLast[m] = time.time()
    first_loop = True
    while stopQue.qsize() is 0:
        try:
            ForceReceive = FingerForceReceiveQue.get(timeout=0.1)
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

        ArucoPose = ForceReceive[6:]
        MachinePosition = pose[0:3]
        Position = MachinePosition - InitPosition
        print(ArucoPose[0:2])
        ArucoPose[1] = -ArucoPose[1]
        ArucoPose[0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]) @ ArucoPose[0:2]
        Finger_tail_end = Position[0:2] + ArucoPose[0:2] * 3.5
        FingerForce[1] = -FingerForce[1]
        FingerForce[0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]]) @ FingerForce[0:2]
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
            if first_loop is True:
                first_loop = False
                Force_init = FingerForce_normal_projection
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
        SGVel[-1] = np.where(np.abs(SGVel[-1]) < gap * 10, 0, np.sign(SGVel[-1]) * (np.abs(SGVel[-1])))

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
