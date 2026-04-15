# pip install gymnasium[classic-control] pygame
import gymnasium as gym
import general_ppo as gppo

# Create an environment
env = gym.make("MountainCarContinuous-v0")
env = gym.make("MountainCarContinuous-v0", render_mode="human")

policy_net = gppo.ActorCriticSimle(2, 0, 1)
policy_net.load_param_from_file("example2_boost.pth")

ppo = gppo.PPOTrainer(policy_net, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4)


def validate(x, v, _p):
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
    return p * _p > 0


for i in range(100):
    memory = gppo.PPOMemory()
    done = False
    total_reward = 0

    # Reset environment to start a new episode
    # [x, velocity] in { [-1.2, 0.7], [-0.07, 0.07] }
    state, info = env.reset()
    while not done:
        # Choose an action
        discrete_action, continuous_action, logprob = ppo.select_action(state)

        p = 0
        if continuous_action[0] > 0:
            p = 1
        else:
            p = -1

        # Take the action and see what happens
        state_new, reward, terminated, truncated, info = env.step([p])
        done = terminated or truncated

        x, v = state
        if validate(x, v, p):
            reward = 1
        else:
            reward = -1

        memory.push(state, discrete_action, continuous_action, logprob, reward, done)
        total_reward += reward
        state = state_new
    print(f"{i}:Episode finished! Total reward: {total_reward}")
    ppo.update(memory.get_memory())


env.close()
policy_net.save_param_to_file("example2_boost.pth")
print("save to file")
