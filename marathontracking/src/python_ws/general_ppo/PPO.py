import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .ActorCriticModule import ActorCriticBase


class PPOMemory:
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def push(
        self,
        state: List[float],
        discrete_action: int,
        continuous_action: List[float],
        logprob: float,
        reward,
        done: bool,
    ):
        self.states.append(state)

        if discrete_action is None:
            self.discrete_actions = None
        else:
            self.discrete_actions.append(discrete_action)

        if continuous_action is None:
            self.continuous_actions = None
        else:
            self.continuous_actions.append(continuous_action)

        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_memory(self):
        return {
            "states": torch.FloatTensor(np.array(self.states, dtype=np.float32)),
            "discrete_actions": (
                torch.LongTensor(self.discrete_actions)
                if self.discrete_actions is not None
                else None
            ),
            "continuous_actions": (
                torch.FloatTensor(np.array(self.continuous_actions, dtype=np.float32))
                if self.continuous_actions is not None
                else None
            ),
            "logprobs": torch.FloatTensor(self.logprobs),
            "rewards": torch.FloatTensor(self.rewards),
            "dones": torch.BoolTensor(self.dones),
        }


class PPOTrainer:
    def __init__(
        self,
        policy: ActorCriticBase,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = policy
        self.policy_old = type(policy)(
            policy.state_dim,
            policy.discrete_action_dim,
            policy.continuous_action_dim,
        )

        self.policy_old.load_state_dict(self.policy.state_dict())
        # Make sure the old policy is not trainable (used only for inference)
        for p in self.policy_old.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # state: numpy or list or tensor, will be converted to Tensor
        # state is shaped [Batch, StateDim], Batch is fixed as 1, so use unsqueeze(0)
        state = torch.FloatTensor(state).unsqueeze(0)

        # discrete_probs   shaped [Batch, N]
        # continuous_mean  shaped [Batch, N]
        # continuous_std   shaped [Batch, N]
        with torch.no_grad():
            discrete_probs, continuous_mean, continuous_std, _ = self.policy_old(state)

        # Sample discrete action
        if self.policy_old.discrete_action_dim > 0:
            discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
            discrete_action = discrete_dist.sample()  # shaped [Batch]
            discrete_logprob = discrete_dist.log_prob(discrete_action)

        # Sample continuous action
        if self.policy_old.continuous_action_dim > 0:
            continuous_dist = torch.distributions.Normal(
                continuous_mean, continuous_std
            )
            continuous_action = continuous_dist.sample()  # shaped [Batch, N]
            continuous_logprob = continuous_dist.log_prob(continuous_action).sum(dim=-1)

        # Combine log probabilities
        if self.policy_old.action_type == 0:
            logprob = discrete_logprob
            return (
                discrete_action.item(),  # just a simple Long number
                None,  # has no data
                logprob.item(),  # just a simple Float number
            )
        elif self.policy_old.action_type == 1:
            logprob = continuous_logprob
            return (
                None,  # just a simple Long number
                continuous_action.numpy().flatten(),  # flattened numpy array
                logprob.item(),  # just a simple Float number
            )
        else:
            logprob = discrete_logprob + continuous_logprob
            return (
                discrete_action.item(),  # just a simple Long number
                continuous_action.numpy().flatten(),  # flattened numpy array
                logprob.item(),  # just a simple Float number
            )

    def update(self, memory):
        states = memory["states"]  # shaped [N, Dim]

        if memory["discrete_actions"] is not None:
            discrete_actions = memory["discrete_actions"].unsqueeze(1)  # shaped [N,1]

        if memory["continuous_actions"] is not None:
            continuous_actions = memory["continuous_actions"]  # shaped [N, Dim]

        old_logprobs = memory["logprobs"].unsqueeze(1)  # shaped [N,1]

        rewards = memory[
            "rewards"
        ]  # shaped [N],  this is used for computing dicounted rewards
        dones = memory[
            "dones"
        ]  # shaped [N],  this is used for computing dicounted rewards

        # Compute discounted rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).unsqueeze(1)  # shaped [N,1]

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for _ in range(self.K_epochs):
            # Get current policy outputs
            discrete_probs, continuous_mean, continuous_std, state_values = self.policy(
                states
            )

            # Discrete action distribution
            if discrete_probs is not None:
                discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
                discrete_logprobs = discrete_dist.log_prob(
                    discrete_actions.squeeze()
                ).unsqueeze(1)

            # Continuous action distribution
            if continuous_mean is not None:
                continuous_dist = torch.distributions.Normal(
                    continuous_mean, continuous_std
                )
                continuous_logprobs = (
                    continuous_dist.log_prob(continuous_actions)
                    .sum(dim=-1)
                    .unsqueeze(1)
                )

            if self.policy.action_type == 0:
                # compute entropy
                entropy = discrete_dist.entropy().mean()
                # Combine log probabilities
                logprobs = discrete_logprobs
            elif self.policy.action_type == 1:
                # compute entropy
                entropy = continuous_dist.entropy().mean()
                # Combine log probabilities
                logprobs = continuous_logprobs
            elif self.policy.action_type == 2:
                # compute entropy
                entropy = (
                    discrete_dist.entropy().mean() + continuous_dist.entropy().mean()
                )
                # Combine log probabilities
                logprobs = discrete_logprobs + continuous_logprobs

            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Compute advantages (detach state values)
            advantages = returns - state_values.detach()
            # Normalize advantages to stabilize training
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            # Policy + value loss (with small clipping on ratios for numerical stability)
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, returns)
            entropy_loss = -0.05 * entropy
            loss = policy_loss + value_loss + entropy_loss

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
