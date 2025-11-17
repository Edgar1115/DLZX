import math
import time
import numpy as np
import torch

class Time:
    "记录时间"
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        "开始计时"
        self.tik = time.time()

    def stop(self):
        "停止计时并记录时间"
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        "计算平均时间"
        return sum(self.times) / len(self.times)

    def sum(self):
        "计算总时间"
        return sum(self.times)

    def cumsum(self):
        "计算累计时间"
        return np.array(self.times).cumsum().tolist()