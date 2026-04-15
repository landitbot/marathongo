# pip install gymnasium[classic-control] pygame
import gymnasium as gym
import general_ppo as gppo

# Create an environment
# env = gym.make("MountainCarContinuous-v0")
env = gym.make("MountainCarContinuous-v0", render_mode="human")

for i in range(1000):
    memory = gppo.PPOMemory()
    done = False
    total_reward = 0

    # Reset environment to start a new episode
    # [x, velocity] in { [-1.2, 0.7], [-0.07, 0.07] }
    state, info = env.reset()
    while not done:
        x, v = state
        p = 0
        if x < -0.5:
            if v < 0:
                p = -1
            else:
                p = 1
        else:
            if v > 0:
                p = 1
            else:
                p = -1

        # Take the action and see what happens
        state_new, reward, terminated, truncated, info = env.step([p])
        done = terminated or truncated
        state = state_new

    print(f"{i}:Episode finished! Total reward: {total_reward}")

env.close()
