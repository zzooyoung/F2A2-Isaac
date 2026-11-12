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
        self.slot_by_step = {}
        self.step_by_slot = {}

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

        step_rows = torch.stack([local[k] for k in STEP_KEYS], dim=1).cpu().tolist()
        step_keys = [tuple(row) for row in step_rows]
        if len(set(step_keys)) != batch_size or any(k in self.slot_by_step for k in step_keys):
            raise ValueError("Duplicate (env_id, episode_id, timestep)")

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

        for slot, key in zip(indices.tolist(), step_keys, strict=True):
            old_key = self.step_by_slot.get(slot)
            if old_key is not None:
                del self.slot_by_step[old_key]
            self.step_by_slot[slot] = key
            self.slot_by_step[key] = slot

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, device="cpu", *, history_steps=0):
        if self.size == 0 or batch_size <= 0 or history_steps < 0:
            raise ValueError("Invalid batch size, history length, or empty Replay")

        indices = torch.randint(self.size, (batch_size,))
        batch = {key: values[indices].to(device) for key, values, in self.storage.items()}

        if history_steps:
            for offset, prefix in ((0, "history"), (1,"next_history")):
                sequences = [
                    self._history(slot, history_steps, offset)
                    for slot in indices.tolist()
                ]
                data, lengths = zip(*sequences)
                batch[prefix] = torch.stack(data).to(device)
                batch[f"{prefix}_lengths"] = torch.tensor(lengths, device=device)

        return batch

    def _pack_peer(self, slots, steps):
        obs = self.storage["peer_obs"]
        action = self.storage["peer_action"]
        padded = obs.new_zeros(steps, obs.shape[-1] + action.shape[-1])
        if slots:
            padded[:len(slots)] = torch.cat((obs[slots], action[slots]), dim = -1)
        return padded, len(slots)

    def _history(self, slot, steps, offset):
        env, episode, timestep = self.step_by_slot[slot]
        newest = timestep + offset - 1
        rows = []

        for t in range(newest, newest - steps, -1):
            previous = self.slot_by_step.get((env, episode, t))
            if previous is None:
                break
            rows.append(previous)

        return self._pack_peer(rows[::-1], steps)

    def character_bank(self, count=32, steps=25, device="cpu"):
        if count <= 0 or steps <= 0:
            raise ValueError("count and steps must be positive")

        sequences = []
        for env, episode, start in self.slot_by_step:
            if start != 0:
                continue

            rows = []
            for t in range(steps):
                slot = self.slot_by_step.get((env, episode, t))
                if slot is None:
                    break
                rows.append(slot)

            last = rows[-1]
            done = self.storage["terminated"][last] | self.storage["truncated"][last]
            if len(rows) == steps or done:
                sequences.append(self._pack_peer(rows, steps))

            if len(sequences) == count:
                break

        if not sequences:
            raise ValueError("Character bank needs a full window or a completed episode")

        data, lengths = zip(*sequences)
        return torch.stack(data).to(device), torch.tensor(lengths, device=device)