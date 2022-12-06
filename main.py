import gym
import numpy as np
import torch

from preprocess import preprocess_obs
from model import DQN

EPSILON = 0.1
N_EPISODES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make("Breakout-v4")
model = DQN(env.action_space.n).to(device)

for n in range(N_EPISODES):

    env.reset()
    terminated = truncated = done = False
    rewards = []

    obs_4 = np.zeros((4, 84, 84), dtype=np.float32)

    # buffer to keep the maximum of last 2 frames
    obs_2_max = np.zeros((2, 84, 84), dtype=np.float32)

    while not done:
        if np.random.uniform() < EPSILON:
            action = env.action_space.sample()
        else:
            action_values = model.forward(torch.from_numpy(obs_4).to(device))
            action = torch.argmax(action_values).item()
        reward = 0
        for i in range(4):
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated or info['lives'] < 5
            if i >= 2:
                obs_2_max[i % 2] = preprocess_obs(obs)
            reward += r
            if done: break
        rewards.append(reward)
        obs = obs_2_max.max(axis=0)

        # push it to the stack of 4 frames
        obs_4 = np.roll(obs_4, shift=-1, axis=0)
        obs_4[-1] = obs

    print(f"Episode {n} done. Total reward: {sum(rewards)}")