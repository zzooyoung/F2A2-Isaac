import torch 
from torch import nn

from actor import squashed_log_prob


def last_valid(gru, data, lengths, initial=None):
    output, _ = gru(data, initial)
    rows = torch.arange(data.shape[0], device=data.device)
    last = output[rows, lengths.clamp_min(1) - 1]
    empty = torch.zeros_like(last) if initial is None else initial[-1]
    return torch.where(lengths[:, None] > 0, last, empty)


class MOA(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12, hidden=64):
        super().__init__()
        self.hidden = hidden
        features = obs_dim + action_dim

        self.character = nn.GRU(features, hidden, num_layers=2, batch_first=True)
        self.mental = nn.GRU(features, hidden, num_layers=2, batch_first=True)
        self.initial_mental = nn.Linear(hidden, 2 * hidden)

        self.natural = nn.Sequential(
            nn.Linear(obs_dim + 2 * hidden, 128), nn.LeakyReLU(),
            nn.Linear(128, 2 * action_dim),
        )
        self.impromptu = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 2 * action_dim),
        )

    def forward(self, query, history, history_lengths, bank):
        past, past_lengths = bank
        past_codes = last_valid(self.character, past, past_lengths)
        character = past_codes.mean(dim=0).expand(query.shape[0], -1)

        initial = self.initial_mental(character).reshape(-1, 2, self.hidden)
        initial = initial.transpose(0, 1).contiguous()
        mental = last_valid(self.mental, history, history_lengths, initial)

        natural = self.natural(torch.cat((query, character, mental), dim=-1))
        impromptu = self.impromptu(query)
        mean, log_std = (0.5 * (natural + impromptu)).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def distribution(self, batch, bank, *, next_step=False):
        query_key = "peer_next_obs" if next_step else "peer_obs"
        history_key = "next_history" if next_step else "history"
        return self(
            batch[query_key], batch[history_key],
            batch[f"{history_key}_lengths"], bank,
        )

    def nll(self, batch, bank):
        mean, log_std = self.distribution(batch, bank)
        return -squashed_log_prob(batch["peer_pre_tanh"], mean, log_std).mean()

    @torch.no_grad()
    def sample_peer(self, batch, bank, *, next_step=False):
        mean, log_std = self.distribution(batch, bank, next_step=next_step)
        pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        return pre_tanh.tanh()