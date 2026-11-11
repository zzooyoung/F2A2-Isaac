import math

import torch
from torch import nn
from torch.nn import functional as F

def squashed_log_prob(pre_tanh, mean, log_std):
    normal = torch.distributions.Normal(mean, log_std.exp())
    gaussian_logp = normal.log_prob(pre_tanh)
    log_tanh_derivative = 2 * (
        math.log(2.0) - pre_tanh - F.softplus(-2*pre_tanh)
    )
    return (gaussian_logp - log_tanh_derivative).sum(dim=-1)

class Actor(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

    def forward(self, obs):
        return self.net(obs).chunk(2, dim=-1)

    @torch.no_grad()
    def collect(self, obs):
        mean, log_std = self(obs)
        pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        return {
            "action": pre_tanh.tanh(),
            "pre_tanh": pre_tanh,
            "behavior_mean": mean,
            "behavior_log_std": log_std,
        }

    def sample_action(self, obs):
        sample = self.collect(obs)
        return sample["action"], sample["pre_tanh"]

    def log_prob(self, obs, pre_tanh):
        mean, log_std = self(obs)
        return squashed_log_prob(pre_tanh, mean, log_std)