# pip install gymnasium[classic-control] pygame
import gymnasium as gym
import general_ppo as gppo

# Create an environment
env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")

policy_net = gppo.ActorCriticSimle(4, 2, 0)
policy_net.load_param_from_file("example1.pth")

ppo = gppo.PPOTrainer(policy_net, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4)


for _ in range(100):
    memory = gppo.PPOMemory()
    done = False
    total_reward = 0

    # Reset environment to start a new episode
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    state, info = env.reset()
    while not done:
        # Choose an action: 0 = push cart left, 1 = push cart right
        discrete_action, continuous_action, logprob = ppo.select_action(state)

        # Take the action and see what happens
        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)
        state_new, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated
        reward = 1.0 - 2.0 * abs(state_new[2]) - 2.0 * abs(state_new[0])
        if terminated:
            reward = -20.0
        if total_reward > 100:
            reward += 2.0
        if total_reward < 50:
            reward *= 2.0  # 如果总奖励较低，增加奖励信号的权重

        memory.push(state, discrete_action, continuous_action, logprob, reward, done)
        total_reward += reward
        state = state_new
    print(f"Episode finished! Total reward: {total_reward}")
    ppo.update(memory.get_memory())


env.close()
policy_net.save_param_to_file("example1.pth")
print("save to file")
