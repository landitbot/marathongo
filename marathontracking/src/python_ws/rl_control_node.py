import general_ppo as gppo
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from RLFuzzyTracking import *
import numpy as np
import threading


"""
Aim: this is a RL-Boosted path tracking algorithm.
Our target is to follow the target path tightly.
"""

"""
states:
- yaw_error[-3.14, 3.14]
- path_curvature[0, 1]
- control var: actual robot linear velocity   (m/s^2)
- control var: actual robot angular velcoity  (rad/s^2)
"""


class RLControlNode:
    def __init__(self):
        self.controller = FuzzyTracking()

        self.ppo_lock = threading.Lock()
        self.policy_net = gppo.ActorCriticSimple(4, 0, 2)
        self.ppo = gppo.PPOTrainer(self.policy_net,
                                   lr=3e-4,
                                   gamma=0.99,
                                   eps_clip=0.2,
                                   K_epochs=4)

        self.memory = gppo.PPOMemory()

        self.policy_net_update = gppo.ActorCriticSimple(4, 0, 2)
        self.ppo_update = gppo.PPOTrainer(self.policy_net_update,
                                          lr=3e-4,
                                          gamma=0.99,
                                          eps_clip=0.2,
                                          K_epochs=4)

        self.last_yaw_error = None
        self.last_curvature = None
        self.last_vel_x = None
        self.last_vel_z = None
        self.last_discrete_action = None
        self.last_continuous_action = None
        self.last_logprob = None
        self.step = 1000
        self.step_count = 0
        self._thread_update = None
        self.updating = False

    def init_ros(self):
        self.suber_tracking_info = rospy.Subscriber(
            "/local_planner/control_info", Float64MultiArray, self.handler_tracking_info,
            queue_size=1)

        self.puber_cmd_vel = rospy.Publisher(
            "/rl_cmd_vel", Twist, latch=False, queue_size=1)

    def pub_cmd_vel(self, vel_x, vel_z):
        msg = Twist()
        msg.linear.x = vel_x
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vel_z
        self.puber_cmd_vel.publish(msg)

    def handler_tracking_info(self, msg: Float64MultiArray):
        target_x = msg.data[0]
        yaw_error = msg.data[1]
        curvatrue = np.clip(msg.data[2], 0.03, 0.95)
        robot_vel_x = msg.data[3]
        robot_vel_z = msg.data[4]
        vel_x, vel_z = self.controller.compute(target_x,
                                               yaw_error,
                                               curvatrue)

        if self.last_yaw_error is None:
            self.last_yaw_error = yaw_error
            self.last_curvature = curvatrue
            self.last_vel_x = robot_vel_x
            self.last_vel_z = robot_vel_z
            return

        state = [self.last_yaw_error,
                 self.last_curvature,
                 self.last_vel_x,
                 self.last_vel_z,
                 ]

        new_state = [
            yaw_error,
            curvatrue,
            robot_vel_x,
            robot_vel_z,
        ]

        done, reward = self.compute_reward(state, new_state)

        if not self.updating:
            self.memory.push(
                state=state,
                discrete_action=self.last_discrete_action,
                continuous_action=self.last_continuous_action,
                logprob=self.last_logprob,
                reward=reward,
                done=done,
            )

            if self.step_count >= self.step:
                self.updating = True
                self._thread_update = threading.Thread(
                    target=self.update, args=(self.memory.get_memory(),), daemon=True)
                self._thread_update.start()

                self.memory = gppo.PPOMemory()
                self.last_yaw_error = None
                self.step_count = 0
            else:
                self.step_count += 1

        with self.ppo_lock:
            discrete_action, continuous_action, logprob =\
                self.ppo.select_action(state)

        self.last_discrete_action = discrete_action
        self.last_continuous_action = continuous_action
        self.last_logprob = logprob
        self.last_vel_x = vel_x + continuous_action[0]
        self.last_vel_z = vel_z + continuous_action[1]

        self.pub_cmd_vel(self.last_vel_x, self.last_vel_z)

    def load_param(self):
        self.policy_net.load_param_from_file("boost1.pth")
        self.policy_net_update.load_param_from_file("boost1.pth")

    def save_param(self):
        self.policy_net.save_param_to_file("boost1.pth")

    def update(self, memories):
        print("Start to update")
        self.ppo_update.update(memories)
        with self.ppo_lock:
            self.policy_net.load_state_dict(
                self.policy_net_update.state_dict())
        self.save_param()
        self.updating = False
        print("Updated")

    def compute_reward(self, old_state, new_state):
        # state = [yaw_error, curvature, robot_vel_x, robot_vel_z]
        yaw_error = new_state[0]
        vel_x = new_state[2]
        vel_z = new_state[3]

        done = False
        reward = 0.0

        # 跟踪精度惩罚
        # 使用平方项或绝对值，误差越大惩罚越重。目的是让误差趋于0。
        # 这里的权重系数 10.0 可以根据实际效果调整
        reward -= 10.0 * (abs(yaw_error) ** 2)

        # 速度激励
        # 鼓励线速度变大，但要设定一个上限防止失控
        # 假设目标最大速度为 1.0 m/s
        reward += 2.0 * vel_x

        # 能量损耗/平稳性惩罚
        # 惩罚过大的角速度，促使路径更平滑，节省电机能量
        reward -= 0.5 * (abs(vel_z) ** 2)

        # 终止条件
        # 成功奖励：如果跟踪得非常紧（例如小于 2 度），给予持续的小奖励
        if abs(yaw_error) < np.deg2rad(2):
            reward += 5.0

        # 失败惩罚：如果偏离过大，强制结束并扣分
        if abs(yaw_error) > np.deg2rad(30):
            done = True
            reward -= 100.0

        return [done, reward]


def main():
    try:
        rospy.init_node("rl_control_node")
        node = RLControlNode()
        rospy.spin()
    except Exception as e:
        print(e)


main()
