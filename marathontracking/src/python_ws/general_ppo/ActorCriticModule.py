import torch
import torch.nn as nn


class ActorCriticBase(nn.Module):
    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_action_dim: int,
    ):
        super(ActorCriticBase, self).__init__()
        self.state_dim = state_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        if discrete_action_dim > 0 and continuous_action_dim > 0:
            self.action_type = 2
        elif discrete_action_dim > 0:
            self.action_type = 0
        elif continuous_action_dim > 0:
            self.action_type = 1
        else:
            raise RuntimeError("Have NOT actions!")

        # actor for discrete actions
        if self.discrete_action_dim > 0:
            self.actor_discrete = nn.Sequential(
                nn.Linear(state_dim, discrete_action_dim),
                nn.Softmax(dim=-1),
            )

        # actor for continuous actions
        if self.continuous_action_dim > 0:
            self.actor_continuous_mean = nn.Sequential(
                nn.Linear(state_dim, continuous_action_dim),
            )
            self.actor_continuous_std = nn.Sequential(
                nn.Linear(state_dim, continuous_action_dim),
                nn.Softplus(),  # Ensure std is positive
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1),
        )

    def forward(self, x):
        if self.discrete_action_dim > 0:
            discrete_probs = self.actor_discrete(x)
        else:
            discrete_probs = None

        if self.continuous_action_dim > 0:
            continuous_mean = self.actor_continuous_mean(x)
            continuous_std = torch.clamp(
                self.actor_continuous_std(x), min=0.01, max=0.5
            )
        else:
            continuous_mean = None
            continuous_std = None
        value = self.critic(x)
        return discrete_probs, continuous_mean, continuous_std, value

    def save_param_to_file(self, filename: str):
        torch.save(self.state_dict(), filename)

    def load_param_from_file(self, filename: str):
        self.load_state_dict(torch.load(filename))


class ActorCriticSimple(ActorCriticBase):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(ActorCriticSimple, self).__init__(
            state_dim,
            discrete_action_dim,
            continuous_action_dim,
        )

        self.hidden_dim = 256

        # actor for discrete actions
        if self.discrete_action_dim > 0:
            self.actor_discrete = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, discrete_action_dim),
                nn.Softmax(dim=-1),
            )

        # actor for continuous actions
        if self.continuous_action_dim > 0:
            self.actor_continuous_mean = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, continuous_action_dim),
                nn.Tanh(),
            )
            self.actor_continuous_std = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, continuous_action_dim),
                nn.Softplus(),  # Ensure std is positive
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )


class ActorCriticTiny(ActorCriticBase):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(ActorCriticTiny, self).__init__(
            state_dim,
            discrete_action_dim,
            continuous_action_dim,
        )

        self.hidden_dim = 32

        # actor for discrete actions
        if self.discrete_action_dim > 0:
            self.actor_discrete = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, discrete_action_dim),
                nn.Softmax(dim=-1),
            )

        # actor for continuous actions
        if self.continuous_action_dim > 0:
            self.actor_continuous_mean = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, continuous_action_dim),
                nn.Tanh(),
            )
            self.actor_continuous_std = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, continuous_action_dim),
                nn.Softplus(),  # Ensure std is positive
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
