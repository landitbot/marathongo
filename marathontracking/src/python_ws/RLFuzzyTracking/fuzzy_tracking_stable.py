import numpy as np
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as plt
import time
# from . import lowpass


"""
Robot Motion Model 1:
input: vel_x, vel_z
output: delta forward, delta angle

Robot Motion Model 2:
input: vel_x, vel_z
output: delta forward, delta horizon
"""


def toDeg(v: float):
    return v * (180.0 / np.pi)


def toRad(v: float):
    return v * (np.pi / 180.0)


def gaussVar38(std: float):
    return math.pow(0.5 * std, 2)


def gaussVar68(std: float):
    return math.pow(std, 2)


def gaussVar95(std: float):
    return math.pow(2.0 * std, 2)


def gaussVar99(std: float):
    return math.pow(3.0 * std, 2)


class VelSmoother:
    def __init__(self, kUp=0.3, kDown=0.5, dt=0.05):
        self.kUp = kUp
        self.kDown = kDown
        self.dt = dt
        self.v = None
        self.last_ts = None

    def compute(self, v):
        if self.v is None:
            self.v = v
            self.last_ts = time.time()
            return self.v
        ts = time.time()
        if ts - self.last_ts < self.dt:
            return self.v
        if v >= self.v:
            self.v = self.v + self.kUp * (v - self.v)
        elif v < self.v:
            self.v = self.v + self.kDown * (v - self.v)
        self.last_ts = ts
        return self.v


class FuzzyTracking:
    def __init__(self):
        self._init_input()
        self._init_output()
        self._init_rules()
        self.smoother = VelSmoother()
        # self.lowpass_curve = lowpass.LowPass(alpha=0.6)
        # self.lowpass_curve_back = lowpass.LowPass(alpha=0.6)
        # self.lowpass_in_theta = lowpass.LowPass(alpha=0.3)

    def _init_input(self):
        self.in_x = ctrl.Antecedent(np.linspace(-0.5, 5.0, 301), "in_x")
        self.in_theta = ctrl.Antecedent(
            np.linspace(-3.14, 3.14, 301), "in_theta")
        self.in_curve = ctrl.Antecedent(
            np.linspace(-0.5, 1.0, 301), "in_curve")
        self.in_curve_back = ctrl.Antecedent(
            np.linspace(-0.5, 1.0, 301), "in_curve_back")

        self.in_x["STOP"] = fuzz.trimf(self.in_x.universe, [-0.5, 1.0, 3.0])
        self.in_x["CLOSE"] = fuzz.trimf(self.in_x.universe, [2.0, 3.0, 4.0])
        self.in_x["FAR"] = fuzz.trapmf(
            self.in_x.universe, [3.0, 4.0, 5.0, 5.0])

        # --------------------------------------------------------

        self.in_theta["ZE"] = fuzz.trapmf(
            self.in_theta.universe, [toRad(-10), 0.0, 0.0, toRad(10)]
        )

        self.in_theta["PS"] = fuzz.trapmf(
            self.in_theta.universe, [toRad(5), toRad(10), toRad(10), toRad(15)]
        )

        self.in_theta["PB"] = fuzz.trapmf(
            self.in_theta.universe, [
                toRad(10), toRad(15), toRad(70), toRad(90)]
        )

        self.in_theta["PEXT"] = fuzz.trapmf(
            self.in_theta.universe, [
                toRad(70), toRad(90), toRad(180), toRad(180)]
        )

        self.in_theta["NS"] = fuzz.trapmf(
            self.in_theta.universe, [
                toRad(-15), toRad(-10), toRad(-10), toRad(-5)]
        )

        self.in_theta["NB"] = fuzz.trapmf(
            self.in_theta.universe, [
                toRad(-90), toRad(-70), toRad(-15), toRad(-10)]
        )

        self.in_theta["NEXT"] = fuzz.trapmf(
            self.in_theta.universe, [
                toRad(-180), toRad(-180), toRad(-90), toRad(-70)]
        )

        self.in_curve["LINE"] = fuzz.trimf(
            self.in_curve.universe, [-0.5, 0.0, 0.1])

        self.in_curve["ROUND"] = fuzz.trimf(
            self.in_curve.universe, [0.05, 0.1, 1.0])

        self.in_curve_back["LINE"] = fuzz.trimf(
            self.in_curve_back.universe, [-0.5, 0.0, 0.1])

        self.in_curve_back["ROUND"] = fuzz.trimf(
            self.in_curve_back.universe, [0.05, 0.1, 1.0])

    def _init_output(self):
        self.out_vel_x = ctrl.Consequent(
            np.linspace(0.0, 1.0, 301), "out_vel_x")

        self.out_vel_z = ctrl.Consequent(
            np.linspace(-3.0, 3.0, 301), "out_vel_z")

        self.out_vel_x["STOP"] = fuzz.trapmf(
            self.out_vel_x.universe, [0.0, 0.0, 0.05, 0.05]
        )

        self.out_vel_x["SLOW"] = fuzz.trapmf(
            self.out_vel_x.universe, [0.1, 0.2, 0.2, 0.3]
        )

        self.out_vel_x["FAST"] = fuzz.trapmf(
            self.out_vel_x.universe, [0.2, 0.7, 0.7, 1.0]
        )

        self.out_vel_x["DASH"] = fuzz.trapmf(
            self.out_vel_x.universe, [0.9, 0.95, 1.0, 1.0]
        )

        # --------------------------------------------------------

        self.out_vel_z["ZE"] = fuzz.trapmf(
            self.out_vel_z.universe, [-1.5, -0.2, 0.2, 1.5]   # -0.0 0.0
        )

        self.out_vel_z["PS"] = fuzz.trapmf(
            self.out_vel_z.universe, [1.0, 1.5, 1.5, 2.0]
        )

        self.out_vel_z["PB"] = fuzz.trapmf(
            self.out_vel_z.universe, [1.5, 2.0, 2.5, 2.5]
        )

        self.out_vel_z["NS"] = fuzz.trapmf(
            self.out_vel_z.universe, [-2.0, -1.5, -1.5, -1.0]
        )

        self.out_vel_z["NB"] = fuzz.trapmf(
            self.out_vel_z.universe, [-2.5, -2.5, -2.0, -1.5]
        )

    def _init_rules(self):
        """
        self.in_x: STOP/CLOSE/FAR
        self.in_theta: ZE/PS/PB/NS/NB/PEXT/NEXT
        self.in_curve: SMALL/MID/LARGE
        self.out_vel_x: STOP/SLOW/FAST/DASH
        self.out_vel_z: ZE/PS/PB/NS/NB
        """
        self.rules = [
            # stop
            ctrl.Rule(
                self.in_x["STOP"] | self.in_theta["PEXT"] | self.in_theta["NEXT"],
                self.out_vel_x["STOP"],
            ),
            # slow
            ctrl.Rule(
                self.in_x["CLOSE"]
                & ~self.in_x["STOP"]
                & (self.in_curve["ROUND"] | self.in_curve_back["ROUND"]),
                self.out_vel_x["SLOW"],
            ),
            # fast
            ctrl.Rule(
                self.in_x["FAR"]
                & ~self.in_x["STOP"]
                & (self.in_curve["ROUND"] | self.in_curve_back["ROUND"]),
                self.out_vel_x["FAST"],
            ),
            # dash
            ctrl.Rule(
                self.in_x["FAR"]
                & ~self.in_x["STOP"]
                & (self.in_curve["LINE"] | self.in_curve_back["LINE"]),
                self.out_vel_x["DASH"],
            ),
            # rotation stop
            ctrl.Rule(self.in_theta["ZE"], self.out_vel_z["ZE"]),
            # rotation slow
            ctrl.Rule(
                self.in_theta["PS"],
                self.out_vel_z["PS"],
            ),
            ctrl.Rule(
                self.in_theta["NS"],
                self.out_vel_z["NS"],
            ),
            # rotation fast
            ctrl.Rule(
                self.in_theta["PB"],
                self.out_vel_z["PB"],
            ),
            ctrl.Rule(
                self.in_theta["NB"],
                self.out_vel_z["NB"],
            ),
            # rotation extrem
            ctrl.Rule(
                self.in_theta["PEXT"],
                self.out_vel_z["PB"],
            ),
            ctrl.Rule(
                self.in_theta["NEXT"],
                self.out_vel_z["NB"],
            ),
        ]

        self.cs = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.cs)

    def view_input(self):
        self.in_x.view()
        self.in_theta.view()
        self.in_curve.view()
        plt.show()

    def view_output(self):
        self.out_vel_x.view()
        self.out_vel_z.view()
        plt.show()

    def view_rules(self):
        print(self.rules[2])

    def compute(self, target_x, close_theta_error, path_curve, path_curve_back=0):
        """
        compute for control variables.

        [target_x]: used to express expected velocity,
                    but could be waken by theta error or path_curve.

        [close_theta_error]: the very close angle error, used to track the path.

        [path_curve]: global velocity referrence.

        :param target_x: the final target point X
        :param close_theta_error: the closest theta angle
        :param path_curve: the whole curvature of the path
        """

        t0 = time.time()
        target_x = np.clip(target_x, 0.03, 4.9)
        close_theta_error = np.clip(close_theta_error, -3.10, 3.10)
        path_curve = np.clip(path_curve, 0.03, 0.95)
        path_curve_back = np.clip(path_curve_back, 0.03, 0.95)
        # close_theta_error = self.lowpass_in_theta.compute(np.clip(close_theta_error, -3.10, 3.10))
        # path_curve = self.lowpass_curve.compute(np.clip(path_curve, 0.0, 0.95))
        # path_curve_back = self.lowpass_curve_back.compute(np.clip(path_curve_back, 0.0, 0.95))

        self.sim.input["in_x"] = target_x
        self.sim.input["in_theta"] = close_theta_error
        self.sim.input["in_curve"] = path_curve
        self.sim.input["in_curve_back"] = path_curve_back
        self.sim.compute()
        output = [
            self.sim.output.get("out_vel_x", 0.0),
            self.sim.output.get("out_vel_z", 0.0),
        ]
        output[0] = np.clip(output[0], 0, 1.0)
        output[1] = np.clip(output[1], -2.6, 2.6)
        if output[0] < 0.05:
            output[0] = 0.0
        output[0] = self.smoother.compute(output[0])
        t1 = time.time()
        print("FuzzyTracking Time:", (t1 - t0) * 1000, "ms")
        return output


if __name__ == "__main__":
    test = FuzzyTracking()
    x = np.linspace(-3.14, +3.14, 100)
    y_velx = [test.compute(5.0, xi, 0.0, 0.0)[0] for xi in x]
    y_velz = [test.compute(5.0, xi, 0.0, 0.0)[1] for xi in x]
    plt.plot(x, y_velx)
    plt.plot(x, y_velz)
    plt.show()

    # x = np.linspace(0, 1.0, 100)
    # y = [test.compute(5.0, 0.0, xi, 0)[0] for xi in x]
    # plt.plot(x, y)
    # plt.show()

    # x = np.linspace(0, 6.0, 100)
    # y = [test.compute(xi, 0, 0.0, 0.0)[0] for xi in x]
    # plt.plot(x, y)
    # plt.show()

    # test.view_input()
    # test.view_output()
    # test.view_rules()
