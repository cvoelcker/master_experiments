import pickle

import gym
import numpy as np
from tqdm import tqdm


def generate_envs_data(env, run_len=100, num_runs=100):
    action_space = env.action_space.n
    all_imgs = np.zeros(
        (num_runs, run_len, *env.observation_space.shape))
    all_actions = np.zeros((num_runs, run_len, action_space))
    all_rewards = np.zeros((num_runs, run_len, 1))
    all_dones = np.zeros((num_runs, run_len, 1))

    # number of sequences
    for run in tqdm(range(num_runs)):
        obs = env.reset()

        # number of steps per sequence
        for t in range(run_len):
            action = env.action_space.sample()
            img, reward, done, _ = env.step(action)
            all_imgs[run, t] = img

            tmp = np.zeros(action_space)
            tmp[action] = 1
            all_actions[run, t - 1] = tmp
            all_rewards[run, t] = reward
            all_dones[run, t] = done

    # save results
    data = dict()
    data['X'] = all_imgs
    data['y'] = all_imgs
    data['action'] = all_actions
    data['reward'] = all_rewards
    data['done'] = all_dones
    data['action_space'] = action_space
    data['coord_lim'] = all_imgs.shape[3]
    return data

if __name__ == '__main__':
    env = gym.make('DemonAttack-v0')
    data = generate_envs_data(env)
    pickle.dump(data, open('./data/demon_attack_train.pkl', 'wb'), protocol=4)
    data = generate_envs_data(env)
    pickle.dump(data, open('./data/demon_attack_test.pkl', 'wb'), protocol=4)
