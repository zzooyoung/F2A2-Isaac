from copy import deepcopy 

import torch

from actor import Actor
from consensus import PDHG
from critic import TwinAttentionCritic
from replay import Replay
from moa import MOA

class AgentLearner:
    def __init__(self, agent_id, initial_critic, *, beta, capacity, device):
        self.agent_id = agent_id
        self.actor = Actor().to(device)
        self.replay = Replay(capacity)
        self.critic = deepcopy(initial_critic).to(device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=0.001
        )
        self.pdhg = PDHG(agent_id, self.critic, beta)
        self.moa = MOA().to(device)
        self.moa_optimizer = torch.optim.Adam(self.moa.parameters(), lr=0.001)

    def moa_step(self, batch, bank):
        self.moa_optimizer.zero_grad(set_to_none=True)
        loss = self.moa.nll(batch, bank)
        loss.backward()
        self.moa_optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def td_target(self, batch, bank, gamma=0.99, temperature=0.01):
        obs = batch["next_obs"]
        action, pre_tanh = self.actor.sample_action(obs)
        own_logp = self.actor.log_prob(obs, pre_tanh)
        peer_action = self.moa.sample_peer(batch, bank, next_step=True)

        q1, q2 = self.target.local_q(
            self.agent_id,
            obs, batch["peer_next_obs"],
            action, peer_action,
        )
        soft_next = torch.minimum(q1, q2) - temperature * own_logp
        bootstrap = torch.where(batch["terminated"], 0.0, soft_next)
        return batch["reward"] + gamma * bootstrap

    def td_critic_step(self, batch, bank):
        target = self.td_target(batch, bank)
        q1, q2 = self.critic.from_replay(batch, self.agent_id)
        td_mse = ((q1 - target).square() + (q2 - target).square()).mean()
        self.critic_step(td_mse)
        return td_mse.detach(), target

    def critic_step(self, objective):
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss = objective + self.pdhg.penalty(self.critic)
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def update_target(self, tau=0.005):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be between 0 and 1")
        for target, local in zip(
            self.target.parameters(), self.critic.parameters(), strict=True
        ):
            target.lerp_(local, tau)

def smoke_consensus():
    torch.manual_seed(7)
    initial = TwinAttentionCritic()
    agents = [
        AgentLearner(i, initial, beta=0.1, capacity=128, device="cpu")
        for i in range(2)
    ]
    obs = torch.randn(8, 2, 48)
    actions = torch.randn(8, 2, 12).tanh()

    for round_id in range(2):
        for agent in agents:
            q1, q2 = agent.critic(obs, actions)
            i = agent.agent_id
            target = float(i + 1)
            objective = (
                (q1[:, i] - target).square() 
                + (q2[:, i] - target).square()
            ).mean()
            agent.critic_step(objective)

        left, right = agents
        a = left.pdhg.snapshot(left.critic)
        b = right.pdhg.snapshot(right.critic)
        left.pdhg.advance(a, b)
        right.pdhg.advance(b, a)

        for name in a.weights:
            torch.testing.assert_close(left.pdhg.z[name], right.pdhg.z[name])
            torch.testing.assert_close(
                left.pdhg.dual[name] + right.pdhg.dual[name],
                torch.zeros_like(left.pdhg.dual[name]),
                rtol=0,
                atol=1e-7,
            )

        gap = torch.stack([
            (a.weights[name] - b.weights[name]).square().sum()
            for name in a.weights
        ]).sum().sqrt()
        assert gap > 0

        for agent in agents:
            agent.update_target()

        print(f"round={round_id} critic_gap={gap.item():.6f}")


if __name__ == "__main__":
    smoke_consensus()
