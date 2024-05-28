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
time.sleep(0.1)
process2.terminate()
if experiment1 is False:
    process1.join()
    time.sleep(0.1)

print(strTime)
from ControlCode.plotForce import main

main(dataDir)
