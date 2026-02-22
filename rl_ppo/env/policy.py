"""Custom MLP policy for the PPO agent."""
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(pi=[256,256], vf=[256,256]),
            activation_fn=nn.ReLU
        )
