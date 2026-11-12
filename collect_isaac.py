import argparse
import importlib.util
import sys
from pathlib import Path
import traceback


def load_bar_task(path):
    spec = importlib.util.spec_from_file_location("f2a2_anymal_bar", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid task source: {path}")
    task  = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = task
    spec.loader.exec_module(task)
    return task


def make_replay_bar(task):
    class ReplayBar(task.AnymalCMultiAgentBar):
        def _get_dones(self):
            timeouts, failures = super()._get_dones()
            return failures, timeouts

        def step(self, actions):
            self.final_observation = None
            self._in_step = True
            try:
                return super().step(actions)
            finally:
                self._in_step = False

        def _reset_idx(self, env_ids):
            if getattr(self, "_in_step", False):
                obs = self._get_observations()
                self.final_observation = (
                    env_ids.clone(),
                    {
                        name: values[env_ids].detach().clone()
                        for name, values in obs.items()
                    },
                )
            return super()._reset_idx(env_ids)

    return ReplayBar

def collect(env, steps, *, actors=None, replays=None):
    import torch

    from actor import Actor
    from critic import TwinAttentionCritic
    from replay import Replay, copy_for_peer

    names = ("robot_0", "robot_1")
    device = env.device
    if actors is None:
        actors = {name: Actor().to(device) for name in names}
    if replays is None:
        replays = {name: Replay(capacity=steps * env.num_envs) for name in names}

    env_id = torch.arange(env.num_envs, device=device)
    episode_id = torch.zeros_like(env_id)
    timestep = torch.zeros_like(env_id)
    final_rows = 0
    obs, _ = env.reset()

    for _ in range(steps):
        current_obs = {name: obs[name].detach().clone() for name in names}
        samples = {name: actors[name].collect(current_obs[name]) for name in names}
        actions = {name: samples[name]["action"] for name in names}
        obs, reward, terminated, truncated, _ = env.step(actions)

        done = terminated[names[0]] | truncated[names[0]]
        peer_done = terminated[names[1]] | truncated[names[1]]
        if not torch.equal(done, peer_done):
            raise RuntimeError("The two robots must reset together")

        next_obs = {name: obs[name].detach().clone() for name in names}
        if done.any():
            if env.final_observation is None:
                raise RuntimeError("Missing observations before reset")
            ids, final = env.final_observation
            if not torch.equal(ids, done.nonzero(as_tuple=True)[0]):
                raise RuntimeError("Final observation rows do not match reset rows")
            for name in names:
                next_obs[name][ids] = final[name]
            final_rows += ids.numel()

        rows = {
            name: {
                "obs": current_obs[name],
                "next_obs": next_obs[name],
                "reward": reward[name],
                "terminated": terminated[name],
                "truncated": truncated[name],
                "env_id": env_id,
                "episode_id": episode_id,
                "timestep": timestep,
                **samples[name],
            }
            for name in names
        }

        replays[names[0]].add(rows[names[0]], copy_for_peer(rows[names[1]]))
        replays[names[1]].add(rows[names[1]], copy_for_peer(rows[names[0]]))
        episode_id = episode_id + done.long()
        timestep = torch.where(done, 0, timestep + 1)

    for i, name in enumerate(names):
        critic = TwinAttentionCritic().to(device)
        batch = replays[name].sample(batch_size=8, device=device)
        with torch.no_grad():
            q1, q2 = critic.from_replay(batch, i)
        if not (q1.isfinite().all() and q2.isfinite().all()):
            raise RuntimeError(f"{name} has nonfinite Q values")
        print(
            f"{name} replay={replays[name].size} "
            f"q1={tuple(q1.shape)} q2={tuple(q2.shape)}"
        )
    print(f"final observation rows={final_rows}")
    return replays

def smoke_td(env, steps):
    import torch

    from critic import TwinAttentionCritic
    from learner import AgentLearner

    names = ("robot_0", "robot_1")
    device = env.device
    initial = TwinAttentionCritic()
    agents = [
        AgentLearner(i, initial, beta=0.1, capacity=steps * env.num_envs, device=device)
        for i in range(2)
    ]
    actors = {name: agent.actor for name, agent in zip(names, agents, strict=True)}
    replays = {name: agent.replay for name, agent in zip(names, agents, strict=True)}

    print("Collecting fixed Character trajectories")
    past = collect(env, max(25, steps), actors=actors)
    banks = [past[name].character_bank(device=device) for name in names]

    print("collecting training transitions")
    collect(env, steps, actors=actors, replays=replays)

    for round_id in range(2):
        batches = [
            agent.replay.sample(16, device=device, history_steps=5)
            for agent in agents
        ]

        for agent, batch, bank in zip(agents, batches, banks, strict=True):
            td_mse, target = agent.td_critic_step(batch, bank)
            terminal = batch["terminated"]
            torch.testing.assert_close(target[terminal], batch["reward"][terminal])

            if not (torch.isfinite(td_mse) and torch.isfinite(target).all()):
                raise RuntimeError("Nonfinite TD update")

            print(
                f"round={round_id} robot_{agent.agent_id} "
                f"target_mean={target.mean().item():.4f} td_mse={td_mse.item():.4f}"
            )

        left, right = agents
        a = left.pdhg.snapshot(left.critic)
        b = right.pdhg.snapshot(right.critic)
        left.pdhg.advance(a, b)
        right.pdhg.advance(b, a)

        for name in left.pdhg.z:
            torch.testing.assert_close(left.pdhg.z[name], right.pdhg.z[name])

        for agent, batch, bank in zip(agents, batches, banks, strict=True):
            agent.update_target()
            agent.moa_step(batch, bank)

def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-source", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--td-smoke", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("--steps and --num-envs must be positive")

    app = AppLauncher(args).app
    try:
        task = load_bar_task(args.task_source.resolve())
        cfg = task.AnymalCMultiAgentFlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = 7
        env = make_replay_bar(task)(cfg=cfg, render_mode=None)
        try:
            if args.td_smoke:
                smoke_td(env, args.steps)
            else:
                collect(env, args.steps)
        finally:
            env.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()



if __name__=="__main__":
    main()

