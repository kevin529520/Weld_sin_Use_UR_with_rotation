import pandas as pd
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
print("1")
import os
print("当前工作目录:", os.getcwd())
config = yaml.load('./controller_config.yaml', Loader=yaml.FullLoader)
print(pd.__version__)
import os
import yaml

# # 获取当前脚本的绝对路径
dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, 'controller_config.yaml')
# with open(config_path, 'r') as file:
config = yaml.load(config_path, Loader=yaml.FullLoader)
#     print("当前工作目录:", config_path)