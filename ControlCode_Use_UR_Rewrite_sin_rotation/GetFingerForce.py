# coding=utf-8
import multiprocessing
import queue

import math

from scipy.spatial.transform import Rotation as R
import pandas as pd
import cv2, os, yaml
from cv2 import aruco

import socket, struct, time
import matplotlib.pyplot as plt
import numpy as np
from queue import LifoQueue
import threading
import multiprocessing as mp


class detector(object):
    def __init__(self, source='./'):
        super().__init__()
        self.source = source

        self._run_flag = True
        # dir_path = os.path.dirname(os.path.realpath(__file__))
# config_path = os.path.join(dir_path, 'controller_config.yaml')

        self.config = yaml.load(open('./ControlCode_Use_UR_Rewrite_sin_rotation/controller_config.yaml'), Loader=yaml.FullLoader)
        # self.config = yaml.load('./controller_config.yaml', Loader=yaml.FullLoader)
        
        self.config = self.config['vision']
        self.camera_matrix = np.matrix(self.config['camera_matrix'])
        self.camera_dist = np.matrix(self.config['camera_dist'])
        self.marker_length = self.config['marker_length']
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.tag_pose_vecs = np.array([[0, 0, 0, 0, 0, 0]])
        self.force_vecs = np.array([[0, 0, 0, 0, 0, 0]])
        self.detection_failure = []
        self.time_cost = []

    def detect(self, color_image):
        # capture from web cam

        gray_ = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray_, -1, kernel)

        current_corners, current_ids, _ = aruco.detectMarkers(gray,
                                                              self.aruco_dict,
                                                              parameters=self.parameters)
        if current_ids is None:
            return None

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(current_corners, 
                                                          self.marker_length, 
                                                          self.camera_matrix, 
                                                          self.camera_dist) 
        
        rr = R.from_rotvec(rvecs[0])
        # 旋转矩阵
        # matrix = rr.as_matrix()
        # rtvecs = rr.apply(tvecs[0][0] * 1000)
        rpy = rr.as_euler('xyz', degrees=True)[0]
        # 欧拉角   
        rpy[-1] = rpy[-1] + 360 * (rpy[-1] < 0)
        rpy[0] = rpy[0] + 360 * (rpy[0] < 0)

        return np.concatenate((tvecs[0][0] * 1000, rpy))
        # 初始 Final Vector: [5.54131605e-01 1.43942161e-01 2.61879730e+01 1.80826534e+02 3.54908876e-01 3.58560626e+02]
        # 形变后 Final Vector: [ -1.91192365   0.54926175  25.51828007 169.97963501  -7.71564783 1.90439425]
class GetFingerForce(object):
    def __init__(self, dataDir):
        self.dataDir = dataDir
        # Camera ready
        for i in range(0, 8):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                print('camera open')
                ret, img = self.cap.read()
                cv2.imshow('2', img)
                cv2.waitKey(1000)
                break
        if not self.cap.isOpened():
            print("no camera")
            exit()
        # self.cap.set(3, 1280)  # 设置分辨率
        # self.cap.set(4, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 330)
        cv2.destroyAllWindows()

        # Aruco ready
        self.detect = detector(dir)
        # detector对象
        tag_pose_vecs = np.zeros([100, 6])
        for i in range(10):
            p = None
            while p is None:
                ret, img = self.cap.read()
                p = self.detect.detect(img)
                # p: [5.54131605e-01 1.43942161e-01 2.61879730e+01 1.80826534e+02
 # 3.54908876e-01 3.58560626e+02]
            p[3] = p[3] + 360 * (p[3] < 0)
            tag_pose_vecs[i, :] = p
        for i in range(100):
            p = None
            while p is None:
                ret, img = self.cap.read()
                p = self.detect.detect(img)
                # 平移分量 + 旋转分量（欧拉角）
            p[3] = p[3] + 360 * (p[3] < 0)
            tag_pose_vecs[i, :] = p
        print("p0 ready")
        self.p0 = np.mean(tag_pose_vecs[:, :], axis=0)
        print("p0", self.p0)
        # 取均值
        self.dpLast = np.zeros([6])
        self.dp = np.zeros([12])
        self.ImgQue = LifoQueue(10000)
        # 后进先出队列，最新的图像会被最先处理
        self.StopImgRead = LifoQueue(1000)

        self.Pose = np.zeros(0)

        self.LastTime = time.time()
        # 记录时间

    def GetImgThread(self):
        while self.StopImgRead.qsize() is 0:
            # 这个队列用于控制图像读取线程的停止，如果队列中有元素（即大小不为0），则停止循环。
            ret, img = self.cap.read()
            self.ImgQue.put(img)
            # LifoQueue(10000)
            time.sleep(0.01)
            # 每次读取后暂停 0.01 秒，这样可以减少 CPU 的使用率，同时给其他线程或进程处理时间。
        return
    # 通过在后台持续读取和存储图像，可以实现实时或接近实时的图像处理应用。

    def getForce(self):
        p = None
        # p = self.detect.detect(img)
        # p: [5.54131605e-01 1.43942161e-01 2.61879730e+01 1.80826534e+02
 # 3.54908876e-01 3.58560626e+02]
        yPredict = np.zeros(6)
        while p is None:
            try:
                img = self.ImgQue.get(timeout=1)
                # LifoQueue(10000)
            except queue.Empty:
                print("Get Img timeout!")
                continue
            p = self.detect.detect(color_image=img)
            # 平移分量 + 旋转分量
            # 两个欧拉角都为正数，Final Vector (in: mm): [ -1.91192365   0.54926175  25.51828007 169.97963501  -7.71564783 1.90439425]
            # self.GrayQue.put(gray)
        p[3] = p[3] + 360 * (p[3] < 0)
        # 将角度转换为正数
        dp = p - self.p0
        # print("dp", dp)
        # p0 [ 3.11083049e-01  2.33430624e-01  2.62575990e+01  1.74294899e+02 -1.32184226e+00  3.58976876e+02]
        # self.p0 = np.mean(tag_pose_vecs[:, :], axis=0)
        # P0是均值，初始化GetFingerForce时计算得到


        yPredict[0] = 6 * dp[0]
        yPredict[1] = 6 * dp[1]
        yPredict[-1] = 0.01487 * dp[-1]
        # print("yPredict", yPredict)
        # x,y 平移 + z 轴偏转

        # Another Finger
        # yPredict_[0] = 2.215 * dp[0] + 0.06409 * dpV[0]
        # yPredict_[1] = 2.027 * dp[1] + 0.05305 * dpV[1]
        # yPredict_[-1] = 0.01485 * dp[-1] + 0.00048 * dpV[-1]

        return yPredict, dp / 1000,img


if __name__ == '__main__':
    fig, ax = plt.subplots()

    dataDir = '../ControlCode/data/' + time.strftime("%H%M%S", time.localtime()) + '/'
    # 类似于 '../ControlCode/data/153025/' 的路径
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    # from ReadOnrobot import OnRobotFT

    self_ = GetFingerForce(dataDir)
    thread4 = threading.Thread(target=self_.GetImgThread, daemon=True)
    # 为持续从摄像头捕获图像，并将其放入一个队列中。通过将这个方法放在一个守护线程（daemon=True）中运行
    thread4.start()
    InitTime = time.time()
    readings = []
    i = 0
    # Set up the plot
    plt.ion()  # Turn on interactive mode for real-time plotting
    try:
        while time.time() - InitTime < 300:
            i = i + 1
            BTime = time.time()

            return_ = self_.getForce()
            # return yPredict, dp / 1000（二维码位姿偏移）,img
            # print(return_)

            if return_ is not None:
                # print("return_", return_)
                yPredict, Pose = return_[:2] # 修改
                # print("yPredict, Pose,", yPredict, Pose) 
                # dp的x y偏移和z轴旋转乘以一个系数   以及二维码位姿偏移dp
                readings.append(yPredict)
                # readings.append(Pose[:3] * 1000)
                forces = np.array(readings).T
                # readings 包含了连续时间点的力和力矩的测量值，转置后的 forces 数组可以让你轻松地访问所有时间点的特定一种力或力矩的值

                # print(np.arange(i))
                # Extract forces in different directions
                # Plot the real-time data
                ax.clear()
                for j, force_direction in enumerate(['x', 'y', 'z', 'mx', 'my', 'mz']):
                # for j, force_direction in enumerate(['x', 'y', 'z']):
                    if j > 2:
                        forces[j] = forces[j] * 10
                    ax.plot(np.arange(i), forces[j], label=force_direction)

                ax.set_title('Real-time Sensor Data')
                ax.set_xlabel('Time')
                ax.set_ylabel('Force')
                ax.legend()
                plt.pause(0.001)  # Pause to allow time for the plot to update
                # print(time.time() - BTime)

            else:
                print('none')
        print("Total iterations:", i)
        print("Average time per iteration:", 1000 * 3 / i)

    except KeyboardInterrupt:
        # Handle keyboard interrupt (e.g., press Ctrl+C to stop the loop)
        pass

    finally:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Display the final plot when the loop ends
    thread4.join()
