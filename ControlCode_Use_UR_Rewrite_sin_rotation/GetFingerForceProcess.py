from matplotlib import pyplot as plt

from GetFingerForce import GetFingerForce as Finger
import multiprocessing as mp
import numpy as np
import time
import threading
import os
import cv2


# 0.0005
def GetFingerForceProcess(dataDir, InitTime, FingerForceReceiveQue=mp.Queue(),
                          stopQue=mp.Queue(), collision=False, CollisionVelQue=mp.Queue()):
    # fig, ax = plt.subplots()

    self = Finger(dataDir)
    thread4 = threading.Thread(target=self.GetImgThread, daemon=True)
    thread4.start()
    OffsetForceRecord = np.zeros((0, 6))
    RecordFingerForceTime = np.zeros(0)

    ArucoPoseRecord = np.zeros((0, 6))
    FingerForceInit = np.zeros((100, 6))
    for i in range(100):
        try:
            return_ = self.getForce()
            if return_ is not None:
                FingerForceInit[i, :], _, img = return_
        except BaseException:
            continue
    photo_array = []
    FingerInitMean = np.mean(FingerForceInit[:, :], axis=0)

    while stopQue.qsize() is 0:
        try:
            # time.sleep(0.01)
            OriginForce, ArucoPose, img = self.getForce()
        except BaseException:
            print("GetFingerForceProcess Error!")
            continue
        OffsetForce = OriginForce - FingerInitMean
        if abs(OffsetForce[0]) > 15 or abs(OffsetForce[1]) > 15 or abs(OffsetForce[-1]) > 10:
            print(OffsetForce)
            FingerForceReceiveQue.put(np.concatenate((OffsetForce, ArucoPose), axis=0))
            print("GetFingerForceProcess Finger Force over above limit")
            stopQue.put(1)
            continue
        if FingerForceReceiveQue.qsize() > 1:
            FingerForceReceiveQue.get()
        FingerForceReceiveQue.put(np.concatenate((OffsetForce, ArucoPose), axis=0))

        RecordFingerForceTime = np.append(RecordFingerForceTime, time.time() - InitTime)
        OffsetForceRecord = np.concatenate((OffsetForceRecord, OffsetForce.reshape(1, 6)), axis=0)
        ArucoPoseRecord = np.concatenate((ArucoPoseRecord, ArucoPose.reshape(1, 6)), axis=0)
        photo_array.append(img)

    np.save(dataDir + 'RecordOffsetForceTime.npy', RecordFingerForceTime, )
    np.save(dataDir + 'ArucoPose.npy', ArucoPoseRecord, )
    np.save(dataDir + 'OffsetForce.npy', OffsetForceRecord)
    if not os.path.exists(dataDir + 'image/'):
        os.makedirs(dataDir + 'image/')
    for idx, image_matrix in enumerate(photo_array):
        # 假设图像矩阵是 BGR 格式，如果是 RGB 格式需要转换成 BGR 格式
        # 如果不是BGR格式，可以不用转换
        bgr_image_matrix = image_matrix

        # 保存图像文件，假设文件名为 "image_{idx}.png"
        cv2.imwrite(dataDir + f"image/image_{idx}.jpg", bgr_image_matrix)
    thread4.join()
    print("FingerForceOver")


def start_process():
    # 创建并启动子进程
    p.start()


if __name__ == '__main__':
    stopQue = mp.Queue()
    p = mp.Process(target=GetFingerForceProcess, args=('./', time.time(), stopQue))
    start_process()
    # 等待30秒后终止子进程
    time.sleep(20)
    stopQue.put(1)
    p.join()