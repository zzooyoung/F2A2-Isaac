import torch 

from actor import Actor, squashed_log_prob
from replay import Replay, copy_for_peer

def main():
    torch.manual_seed(7)

    actors = [Actor(), Actor()]
    replays = [Replay(capacity=128), Replay(capacity=128)]

    num_envs = 2
    env_id = torch.arange(num_envs)
    episode_id = torch.zeros(num_envs, dtype=torch.long)
    timestep = torch.zeros_like(episode_id)
    obs = [torch.randn(num_envs, 48) for _ in actors]

    for step in range(6):
        samples = [actor.collect(local_obs) for actor, local_obs in zip(actors, obs)]

        # 가짜 환경의 step 결과, Isaac Lab 연결 시 이 부분이 env.step()이 된다. 
        next_obs = [x + 0.01 * torch.randn_like(x) for x in obs]
        reward = torch.ones(num_envs)
        terminated = torch.tensor([False, step == 2])
        truncated = torch.tensor([step == 4, False])
        done = terminated | truncated

        transitions = [
            {
                "obs": obs[i],
                "next_obs": next_obs[i],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "env_id": env_id,
                "episode_id": episode_id,
                "timestep": timestep,
                **samples[i],
            }
            for i in range(2)
        ]

        replays[0].add(transitions[0], copy_for_peer(transitions[1]))
        replays[1].add(transitions[1], copy_for_peer(transitions[0]))

        obs = [
            torch.where(done[:, None], torch.zeros_like(x), x)
            for x in next_obs
        ]
        episode_id = episode_id + done.long()
        timestep = torch.where(done, 0, timestep + 1)

    for i, (actor, replay) in enumerate(zip(actors, replays)):
        batch = replay.sample(batch_size=8)
        with torch.no_grad():
            old_logp = squashed_log_prob(
                batch["pre_tanh"],
                batch["behavior_mean"],
                batch["behavior_log_std"],
            )
            current_logp = actor.log_prob(batch["obs"], batch["pre_tanh"])
            ratio = (current_logp - old_logp).clamp(max=0.0).exp()

        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        print(
            f"robot_{i} | replay={replay.size} | "
            f"obs={tuple(batch['obs'].shape)} | "
            f"peer_obs={tuple(batch['peer_obs'].shape)} | "
            f"mean ratio={ratio.mean().item():.3f}"
        )

if __name__ == "__main__":
    main()

