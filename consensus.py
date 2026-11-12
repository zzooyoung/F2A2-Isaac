from dataclasses import dataclass 

import torch


def frozen_weights(critic):
    return {name: p.detach().clone() for name, p in critic.named_parameters()}


@dataclass(frozen=True)
class CriticSnapshot:
    sender: int
    round_id: int
    weights: dict[str, torch.Tensor]


class PDHG:
    def __init__(self, agent_id, critic, beta):
        if agent_id not in (0, 1) or beta <= 0:
            raise ValueError("Expected agent - or 1 and a positive beta")

        self.agent_id = agent_id
        self.beta = beta
        self.round_id = 0
        self.z = frozen_weights(critic)
        self.dual = {
            name: torch.zeros_like(value) for name, value in self.z.items()
        }

    def penalty(self, critic):
        terms = []
        for name, p in critic.named_parameters():
            residual = p -self.z[name]
            terms.append(
                (self.dual[name] * residual).sum()
                + 0.5 * self.beta * residual.square().sum()
            )
        return torch.stack(terms).sum()

    def snapshot(self, critic):
        return CriticSnapshot(
            self.agent_id, self.round_id, frozen_weights(critic)
        )

    @torch.no_grad()
    def advance(self, own, peer):
        if own.sender != self.agent_id or peer.sender != 1 - self.agent_id:
            raise ValueError("Unexpected snapshot sender")
        if own.round_id != self.round_id or peer.round_id != self.round_id:
            raise ValueError("Snapshots must belong to the current round")

        for packet in (own, peer):
            if packet.weights.keys() != self.z.keys():
                raise ValueError("Critic parameter names do not match")
            for name, value in packet.weights.items():
                ref = self.z[name]
                if (value.shape, value.dtype, value.device) != (
                    ref.shape, ref.dtype, ref.device
                ):
                    raise ValueError(f"Critic parameter mismatch: {name}")

        new_z = {
            name: 0.5 * (own.weights[name] + peer.weights[name])
            for name in self.z
        }
        for name, value in new_z.items():
            self.dual[name].add_(self.beta * (own.weights[name] - value))

        self.z = new_z
        self.round_id += 1