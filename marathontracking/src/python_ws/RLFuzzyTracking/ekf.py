import numpy as np
import time
from abc import ABC, abstractmethod


class EKF(ABC):
    """
    扩展卡尔曼滤波器基类
    状态维度 n，观测维度 m
    """

    def __init__(self, n: int, m: int):
        """
        n : 状态维度
        m : 观测维度
        """
        self.n = n
        self.m = m

        # 状态向量及其协方差
        self.x = np.zeros((n, 1))
        self.P = np.eye(n) * 1e6

        # 过程噪声和观测噪声协方差（子类可在构造函数里覆盖）
        self.Q = np.eye(n)
        self.R = np.eye(m)

        # 时间计时
        self.ts = None

    # ---------------- 需要子类实现的 4 个接口 ----------------
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """运动学模型：x_k = f(x_{k-1}, u, dt)"""
        pass

    @abstractmethod
    def F_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """df/dx 的雅可比矩阵 (n×n)"""
        pass

    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """观测模型：z = h(x)"""
        pass

    @abstractmethod
    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """dh/dx 的雅可比矩阵 (m×n)"""
        pass

    # -------------------------------------------------------

    def predict(self, u: np.ndarray, dt: float = None):
        """先验预测"""
        if dt is None:
            if self.ts is None:
                dt = 1e-6
            else:
                dt = time.time() - self.ts
            self.ts = time.time()

        # 1. 状态预测
        self.x = self.f(self.x, u, dt)
        # 2. 协方差预测
        F = self.F_jacobian(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        """后验更新"""
        z_pred = self.h(self.x)
        y = z - z_pred  # 新息
        H = self.H_jacobian(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

        self.x += K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def reset(self):
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n) * 1e7

    # ---------------- 辅助工具 ----------------
    def set_state(self, x: np.ndarray, P: np.ndarray = None):
        self.x = x.reshape(self.n, 1)
        if P is not None:
            self.P = P

    def get_state(self):
        return self.x.copy(), self.P.copy()


class UniformSpeedEKF(EKF):
    def __init__(self):
        """
        x=[x, y, z, vx, vy, vz]
        z=[x, y, z]
        """
        super().__init__(6, 3)

        self.Q = (
            np.diag(
                [
                    0.1,  # x
                    0.1,  # y
                    0.1,  # z
                    0.1,  # vx
                    0.1,  # vy
                    0.1,  # vz
                ]
            )
            ** 2
        )

        self.R = (
            np.diag(
                [
                    0.15,  # x
                    0.15,  # y
                    0.15,  # z
                ]
            )
            ** 2
        )

    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """运动学模型：x_k = f(x_{k-1}, u, dt)"""
        F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        )
        return F @ x

    def F_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        )

    def h(self, x: np.ndarray) -> np.ndarray:
        """观测模型：z = h(x)"""
        H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
        )
        return H @ x

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """dh/dx 的雅可比矩阵 (m×n)"""
        return np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
        )
