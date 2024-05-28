# MIT License.
# Copyright (c) 2021 by BioicDL. All rights reserved.
# Created by LiuXb & JieYu on 2021/11/28
# -*- coding:utf-8 -*-

"""
@Modified:
@Description:
"""
import numpy as np
from ControlCode.URRtde import URControl as UR
# from URRtde import URControl as UR
import time
import queue
import math
import multiprocessing as mp
from scipy.spatial.transform import Rotation as Rt


def judgeWall(pos):
    print(pos[1])
    if pos[1] > 0.14:
        return True
    else:
        return False


class RobotClient(object):
    def __init__(self):
        self.robot = UR().get_robot()

    def Speed(self, speedX, speedY, speedZ, speedMx, speedMy, speedMz, acc=0.25):  # pose angle degree
        if abs(speedX) > 0.3 or abs(speedY) > 0.3 or abs(speedZ) > 0.3:
            self.close()
            raise RuntimeError('vel is too big')
        Mxyz = False
        if Mxyz:
            speed = [0, 0, 0, speedMx, speedMy, speedMz]
            print(speedMy, speedMz)
        else:
            speed = [speedX, speedY, speedZ, 0, 0, speedMz]
        # print(speed)
        # acc = max(abs(np.array([speedX, speedY, speedZ])).max() * 2, 0.025)
        acc = 0.25
        self.robot.control_c.speedL(speed, acceleration=acc, time=0)

    def Move(self, init_pose, tcp):  # pose angle degree
        self.robot.control_c.moveL((init_pose[0], init_pose[1], init_pose[2], tcp[0], tcp[1], tcp[2]), speed=0.05,
                                   acceleration=0.1, asynchronous=False)

    def moveJoint(self, joint):
        self.robot.control_c.moveJ(joint, speed=0.2)

    def ReadMachineSpeed(self):
        velocity = self.robot.receive_r.getTargetTCPSpeed()
        vel = np.array([velocity[0], velocity[1], velocity[2]])
        return vel

    def ReadMachinePosition(self):
        position = np.array(self.robot.receive_r.getActualTCPPose())
        return position

    def ReadTCPForce(self):
        force = self.robot.receive_r.getActualTCPForce()
        force = np.array(force)
        return force

    def virtualWall(self, q, callback):
        while True:
            pos = self.ReadMachinePosition()
            if judgeWall(pos):
                q.put(1000)
            else:
                q.put(30)
            callback.get()

    def close(self):
        speed = [0, 0, 0, 0, 0, 0]
        self.robot.control_c.speedL(speed)
        time.sleep(0.1)
        self.robot.control_c.disconnect()
        # self.robot.CloseModbusTCPConnection()




if __name__ == "__main__":
    self = RobotClient()
    # move X
    self.ReadMachinePosition()
    self.ReadMachineSpeed()
    self.ReadTCPForce()
    # self.Speed(0, 0, -0.01, 0.4, 0.4, 0.4)  # z轴向上为正
    time.sleep(1)
    self.close()
