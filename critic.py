import torch
from torch import nn


class AttentionQ(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12, hidden=128, heads=4):
        super().__init__()
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.LeakyReLU())
            for _ in range(2)
        )
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.q_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1)
            )
            for _ in range(2)
        )
        self.register_buffer("self_mask", torch.eye(2, dtype=torch.bool))

    def forward(self, obs, action):
        obs_action = torch.cat((obs, action), dim = -1)
        encoded = torch.stack(
            [encoder(obs_action[:, i]) for i, encoder in enumerate(self.encoders)],
            dim = 1,
        )
        context, _ = self.attention(
            encoded, encoded, encoded,
            attn_mask=self.self_mask,
            need_weights=False,
        )
        features = torch.cat((encoded, context), dim = -1)
        return torch.cat(
            [head(features[:, i]) for i, head in enumerate(self.q_heads)],
            dim = -1,
        )

class TwinAttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = AttentionQ()
        self.q2 = AttentionQ()

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def local_q(self, agent_id, own_obs, peer_obs, own_action, peer_action):
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")

        def ordered(local, peer):
            pair = (local, peer) if agent_id == 0 else (peer, local)
            return torch.stack(pair, dim=1)

        q1, q2 = self(
            ordered(own_obs, peer_obs),
            ordered(own_action, peer_action),
        )
        return q1[:, agent_id], q2[:, agent_id]

    def from_replay(self, batch, agent_id):
        return self.local_q(
            agent_id,
            batch["obs"], batch["peer_obs"],
            batch["action"], batch["peer_action"],
        )