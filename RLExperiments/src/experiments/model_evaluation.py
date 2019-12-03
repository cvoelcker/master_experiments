import numpy as np
import torch

from tqdm import tqdm


def evaluate_model(model, env, eval_runs=100):
    done = np.array([False] * env.num_envs)
    total_rewards = []
    steps = 0
    for i in tqdm(list(range(eval_runs))):
        env.reset()
        done = np.array([False] * env.num_envs)
        rewards = []
        iterator = tqdm(range(10000))
        for env_steps in iterator:
            state = env.stacked_obs
            actor, _ = model.eval_step(state)
            action = actor.sample()
            _, reward, _done, info = env.step(action.unsqueeze(1))
            done = np.logical_or(done, _done)
            mask = np.array([0.0 if d else 1.0 for d in done])
            rewards.append(reward.detach().numpy().squeeze() * mask)
            env_steps += 1
            if np.all(done):
                iterator.close()
                break
        steps += env_steps
        total_rewards.append(np.sum(np.array(rewards), 0))
    return np.array(total_rewards), steps

