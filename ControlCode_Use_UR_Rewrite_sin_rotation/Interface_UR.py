import os
import time
import multiprocessing as mp
import RockControlUsingFinger as Rock
from GetFingerForceProcess import GetFingerForceProcess
from curl_path_1_2 import get_curl_path
import tkinter as tk
from tkinter import messagebox

gun_path_spline, gun_points, finger_points_spline, finger_points = get_curl_path()

strTime = time.strftime("%m%d%H%M%S", time.localtime())
dataDir = '../ControlCode/data/' + strTime + '/'
try:
    os.mkdir(dataDir)
except BaseException:
    pass
 
experiment1 = False  # False代表末端运动

stopQue = mp.Queue()
FingerForceReceiveQue = mp.Queue()
InitTime = time.time()

process1 = mp.Process(target=Rock.AdmittanceControl,
                      args=(dataDir, InitTime, gun_path_spline, gun_points, finger_points_spline, finger_points,
                            FingerForceReceiveQue, stopQue, False, True,),
                      daemon=False)

process2 = mp.Process(target=GetFingerForceProcess, args=(dataDir, InitTime, FingerForceReceiveQue,
                                                          stopQue), daemon=False)

process2.start()
if experiment1 is False:
    process1.start()


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stopQue.put(1)
        root.destroy()


root = tk.Tk()
root.title("Parameter Setting")
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

process2.join(timeout=1)
# 调用 process2.join(timeout=1)，表示主进程将等待 process2 结束，最多等待 1 秒。
# 如果 process2 在 1 秒内结束，主进程将继续执行。如果超时，主进程将继续执行而不等待 process2。
time.sleep(0.1)
# 暂停主进程 0.1 秒。这通常用于确保某些操作之间有足够的时间间隔。
process2.terminate()
# 强制终止 process2 进程。
# 这一操作在 process2 仍在运行时很有用，确保进程被正确终止。
if experiment1 is False:
    process1.join()
    # process1.join() 等待 process1 结束。
    time.sleep(0.1)

print(strTime)
from ControlCode.plotForce import main

main(dataDir)
