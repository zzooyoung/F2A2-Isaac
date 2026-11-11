import torch


STEP_KEYS = ("env_id", "episode_id", "timestep")
PEER_KEYS = ("obs", "action", "pre_tanh", "next_obs", *STEP_KEYS)

def copy_for_peer(transition):
    return {key: transition[key].detach().clone() for key in PEER_KEYS}

class Replay:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.storage = {}

    def add(self, local, peer):
        for key in STEP_KEYS:
            if not torch.equal(local[key], peer[key]):
                raise ValueError(f"Local and peer transitions disagree on {key}")

        fields = dict(local)
        for key in PEER_KEYS:
            if key not in STEP_KEYS:
                fields[f"peer_{key}"] = peer[key]

        batch_size = local["obs"].shape[0]
        if not 0 < batch_size <= self.capacity:
            raise ValueError("Invalid transition batch size")

        if not self.storage:
            self.storage = {
                key: torch.empty(
                    (self.capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device="cpu",
                )
                for key, value in fields.items()
            }

        indices = (self.position + torch.arange(batch_size)) % self.capacity
        for key, value in fields.items():
            self.storage[key][indices] = value.detach().cpu()

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if self.size == 0 or batch_size <= 0:
            raise ValueError("Replay must contain data and batch_size must be positive")
        indices = torch.randint(self.size, (batch_size,))
        return {
            key: values[indices].to(device)
            for key, values in self.storage.items()
        }
        