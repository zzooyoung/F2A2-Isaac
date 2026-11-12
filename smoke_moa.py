import torch

from critic import TwinAttentionCritic
from learner import AgentLearner
from smoke_collect import main as collect_fake



def main():
    past_replays = collect_fake(seed=7)
    train_replays = collect_fake(seed=8)
    initial = TwinAttentionCritic()

    for i in range(2):
        agent = AgentLearner(i, initial, beta=0.1, capacity=128, device="cpu")
        bank = past_replays[i].character_bank()
        batch = train_replays[i].sample(16, history_steps=5)

        with torch.no_grad():
            before = agent.moa.nll(batch, bank).item()

        for _ in range(20):
            agent.moa_step(batch, bank)

        with torch.no_grad():
            after = agent.moa.nll(batch, bank).item()

        actions = agent.moa.sample_peer(batch, bank, next_step=True)
        assert after < before
        assert actions.shape == (16, 12)
        assert torch.isfinite(actions).all()
        print(f"robot_{i} MOA NLL {before:.3f} -> {after:.3f} | actions={tuple(actions.shape)}")



if __name__ == "__main__":
    main()