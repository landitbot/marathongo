import numpy as np
import time
import math
from matplotlib import pyplot as plt


class Test:
    def __init__(self):
        pass

    def smooth_stop(self, start_vel=0.8, interval=0.05, stop_duration=1.0):
        x = []
        y = []
        t = 0
        for i in np.linspace(2.5, 0.0, int(stop_duration / interval)):
            vel = math.tanh(i)*start_vel
            x.append(t * interval)
            t += 1
            y.append(vel)
            time.sleep(interval)
        plt.xlabel("time(sec)")
        plt.ylabel("vel(m/s)")
        plt.plot(x, y)
        plt.show()


t = Test()
t.smooth_stop()
