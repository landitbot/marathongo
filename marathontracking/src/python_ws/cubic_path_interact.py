
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    # 将坐标系平移到以x0为原点，简化计算
    h = x1 - x0
    
    if abs(h) < 1e-10:
        return np.full_like(x, y0)
    
    # 计算三次多项式系数 (使用Hermite插值形式)
    # y = y0 + m0*(x-x0) + (3*(y1-y0)/h - 2*m0 - m1)*(x-x0)²/h + (m0 + m1 - 2*(y1-y0)/h)*(x-x0)³/h²
    
    t = (x - x0) / h
    
    # Hermite基函数
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    y = y0 * h00 + h * m0 * h10 + y1 * h01 + h * m1 * h11
    
    return y

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# 初始参数
x0_init, y0_init = 1, 2
x1_init, y1_init = 8, 5
m0_init, m1_init = 0.5, -0.3

# 生成曲线数据
x = np.linspace(0, 10, 500)
y = cubic_curve(x, x0_init, y0_init, x1_init, y1_init, m0_init, m1_init)

# 绘制曲线
line, = ax.plot(x, y, 'b-', linewidth=2.5, label='Cubic Curve')

# 绘制起始点和终点
point0, = ax.plot(x0_init, y0_init, 'ro', markersize=10, label='Start Point')
point1, = ax.plot(x1_init, y1_init, 'go', markersize=10, label='End Point')

# 绘制切线（用于可视化斜率）
tangent_length = 1.5
t0_x = np.array([x0_init - tangent_length/2, x0_init + tangent_length/2])
t0_y = y0_init + m0_init * (t0_x - x0_init)
tangent0, = ax.plot(t0_x, t0_y, 'r--', alpha=0.6, linewidth=1.5)

t1_x = np.array([x1_init - tangent_length/2, x1_init + tangent_length/2])
t1_y = y1_init + m1_init * (t1_x - x1_init)
tangent1, = ax.plot(t1_x, t1_y, 'g--', alpha=0.6, linewidth=1.5)

# 设置坐标轴
ax.set_xlim(0, 10)
ax.set_ylim(-1, 8)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Interactive Cubic Polynomial Curve\n(Control Start/End Points and Slopes)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 创建滑块
axcolor = 'lightgoldenrodyellow'

# 起始点X
ax_x0 = plt.axes([0.15, 0.25, 0.3, 0.03], facecolor=axcolor)
s_x0 = Slider(ax_x0, 'Start X', 0.5, 4.5, valinit=x0_init)

# 起始点Y  
ax_y0 = plt.axes([0.15, 0.20, 0.3, 0.03], facecolor=axcolor)
s_y0 = Slider(ax_y0, 'Start Y', 0, 7, valinit=y0_init)

# 起始斜率
ax_m0 = plt.axes([0.15, 0.15, 0.3, 0.03], facecolor=axcolor)
s_m0 = Slider(ax_m0, 'Start Slope', -3, 3, valinit=m0_init)

# 终点X
ax_x1 = plt.axes([0.6, 0.25, 0.3, 0.03], facecolor=axcolor)
s_x1 = Slider(ax_x1, 'End X', 5.5, 9.5, valinit=x1_init)

# 终点Y
ax_y1 = plt.axes([0.6, 0.20, 0.3, 0.03], facecolor=axcolor)
s_y1 = Slider(ax_y1, 'End Y', 0, 7, valinit=y1_init)

# 终点斜率
ax_m1 = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=axcolor)
s_m1 = Slider(ax_m1, 'End Slope', -3, 3, valinit=m1_init)

# 更新函数
def update(val):
    x0 = s_x0.val
    y0 = s_y0.val
    x1 = s_x1.val
    y1 = s_y1.val
    m0 = s_m0.val
    m1 = s_m1.val
    
    # 确保x1 > x0
    if x1 <= x0:
        x1 = x0 + 0.5
        s_x1.set_val(x1)
    
    # 更新曲线
    y_new = cubic_curve(x, x0, y0, x1, y1, m0, m1)
    line.set_ydata(y_new)
    
    # 更新点位置
    point0.set_data([x0], [y0])
    point1.set_data([x1], [y1])
    
    # 更新切线
    t0_x = np.array([x0 - tangent_length/2, x0 + tangent_length/2])
    t0_y = y0 + m0 * (t0_x - x0)
    tangent0.set_data(t0_x, t0_y)
    
    t1_x = np.array([x1 - tangent_length/2, x1 + tangent_length/2])
    t1_y = y1 + m1 * (t1_x - x1)
    tangent1.set_data(t1_x, t1_y)
    
    fig.canvas.draw_idle()

# 绑定滑块事件
s_x0.on_changed(update)
s_y0.on_changed(update)
s_m0.on_changed(update)
s_x1.on_changed(update)
s_y1.on_changed(update)
s_m1.on_changed(update)

plt.show()
