import numpy as np
import matplotlib.pyplot as plt
import os

# 三次多项式曲线生成函数


def cubic_curve(x, x0, y0, x1, y1, m0, m1):
    """
    生成通过两点(x0,y0)和(x1,y1)，且在这两点斜率分别为m0和m1的三次多项式曲线

    三次多项式: y = ax³ + bx² + cx + d
    约束条件:
    - f(x0) = y0
    - f(x1) = y1
    - f'(x0) = m0
    - f'(x1) = m1
    """
    h = x1 - x0  # 将坐标系平移到以x0为原点
    if abs(h) < 1e-10:
        return np.full_like(x, y0)
    t = (x - x0) / h
    # Hermite基函数
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    y = y0 * h00 + h * m0 * h10 + y1 * h01 + h * m1 * h11
    return y


outputdir = '/opt/marathon_ws/pathes'
save_to_file = True


def save_path(name, x, y, outputdir="."):
    global save_to_file
    if not save_to_file:
        return
    pathfile = os.path.join(outputdir, name)
    with open(pathfile, mode='w+') as f:
        for i in range(len(x)):
            f.write(f'{x[i]},{y[i]},0.0,')


def get_path_name(A, B):
    return f'{A}{B:03d}.path'


def generate():
    y0 = np.linspace(-3.0, 3.0, 30)
    for i in y0:
        x = np.linspace(0, 8.0, 100)
        y = cubic_curve(x, 0, i, 8.0, i, 0, 0)
        # save_path(get_path_name('A', i), x, y,
        #           outputdir)
        plt.plot(x, y)


generate()
plt.show()
