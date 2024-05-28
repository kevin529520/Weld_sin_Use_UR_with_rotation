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
        self.config = yaml.load(open('./controller_config.yaml'), Loader=yaml.FullLoader)
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
        # matrix = rr.as_matrix()
        # rtvecs = rr.apply(tvecs[0][0] * 1000)
        rpy = rr.as_euler('xyz', degrees=True)[0]
        rpy[-1] = rpy[-1] + 360 * (rpy[-1] < 0)
        rpy[0] = rpy[0] + 360 * (rpy[0] < 0)

        return np.concatenate((tvecs[0][0] * 1000, rpy))


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
        tag_pose_vecs = np.zeros([100, 6])
        for i in range(10):
            p = None
            while p is None:
                ret, img = self.cap.read()
                p = self.detect.detect(img)
            p[3] = p[3] + 360 * (p[3] < 0)
            tag_pose_vecs[i, :] = p
        for i in range(100):
            p = None
            while p is None:
                ret, img = self.cap.read()
                p = self.detect.detect(img)
            p[3] = p[3] + 360 * (p[3] < 0)
            tag_pose_vecs[i, :] = p
        print("p0 ready")
        self.p0 = np.mean(tag_pose_vecs[:, :], axis=0)
        self.dpLast = np.zeros([6])
        self.dp = np.zeros([12])
        self.ImgQue = LifoQueue(10000)
        self.StopImgRead = LifoQueue(1000)

        self.Pose = np.zeros(0)

        self.LastTime = time.time()

    def GetImgThread(self):
        while self.StopImgRead.qsize() is 0:
            ret, img = self.cap.read()
            self.ImgQue.put(img)
            time.sleep(0.01)
        return

    def getForce(self):
        p = None
        yPredict = np.zeros(6)
        while p is None:
            try:
                img = self.ImgQue.get(timeout=1)
            except queue.Empty:
                print("Get Img timeout!")
                continue
            p = self.detect.detect(color_image=img)
            # self.GrayQue.put(gray)
        p[3] = p[3] + 360 * (p[3] < 0)
        dp = p - self.p0

        yPredict[0] = 6 * dp[0]
        yPredict[1] = 6 * dp[1]
        yPredict[-1] = 0.01487 * dp[-1]
        # Another Finger
        # yPredict_[0] = 2.215 * dp[0] + 0.06409 * dpV[0]
        # yPredict_[1] = 2.027 * dp[1] + 0.05305 * dpV[1]
        # yPredict_[-1] = 0.01485 * dp[-1] + 0.00048 * dpV[-1]

        return yPredict, dp / 1000,img


if __name__ == '__main__':
    fig, ax = plt.subplots()

    dataDir = '../ControlCode/data/' + time.strftime("%H%M%S", time.localtime()) + '/'
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    # from ReadOnrobot import OnRobotFT

    self_ = GetFingerForce(dataDir)
    thread4 = threading.Thread(target=self_.GetImgThread, daemon=True)
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
            # print(return_)

            if return_ is not None:
                yPredict, Pose, = return_
                readings.append(yPredict)
                forces = np.array(readings).T

                # print(np.arange(i))
                # Extract forces in different directions
                # Plot the real-time data
                ax.clear()
                for j, force_direction in enumerate(['x', 'y', 'z', 'mx', 'my', 'mz']):
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
