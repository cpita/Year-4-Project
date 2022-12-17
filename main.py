import gym
import numpy as np
import torch
import random
from collections import deque
from datetime import datetime as dt

from preprocess import preprocess_obs
from model import DQN

N_FRAMES_TOTAL = 10 ** 8
N_FRAMES_EPS_DECAY = 10 ** 7
REPLAY_BUFFER_SIZE = 10 ** 6
GAMMA = 0.99
MINI_BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make("Breakout-v4")
model = DQN(env.action_space.n).to(device)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
optimizer = torch.optim.RMSprop(model.parameters())
loss_function = torch.nn.MSELoss()
n_frames = 0
eval = True

while n_frames < N_FRAMES_TOTAL:

    env.reset()
    terminated = truncated = done = False

    phi_t = np.zeros((4, 84, 84), dtype=np.float32)

    # buffer to keep the maximum of last 2 frames
    obs_2_max = np.zeros((2, 84, 84), dtype=np.float32)

    while not done:

        # Choose action according to eps-greedy policy
        eps_decay = (N_FRAMES_EPS_DECAY - min(n_frames, N_FRAMES_EPS_DECAY)) / N_FRAMES_EPS_DECAY
        if np.random.uniform() < 0.1 * (1 - eps_decay) + 1 * eps_decay:
            a_t = env.action_space.sample()
        else:
            action_values = model.forward(torch.from_numpy(phi_t).to(device))
            a_t = torch.argmax(action_values).item()

        # Play for 4 frames using that action
        r_t = 0
        for i in range(4):
            obs, r, terminated, truncated, info = env.step(a_t)
            done = terminated or truncated or info['lives'] < 5
            if i >= 2:
                obs_2_max[i % 2] = preprocess_obs(obs)
            r_t += r
            if done: break

        # Store the state transition, action and reward into the replay buffer
        x_t_plus1 = obs_2_max.max(axis=0)
        phi_t_plus1 = np.roll(phi_t, shift=-1, axis=0)
        phi_t_plus1[-1] = x_t_plus1
        replay_buffer.append((phi_t, a_t, r_t, phi_t_plus1, done))

        # Take a random mini-batch from the replay buffer and calculate labels
        mini_batch = random.choices(replay_buffer, k=MINI_BATCH_SIZE)
        phi_j, a_j, r_j, phi_j_plus1, is_terminal_j = [], [], [], [], []
        for phi, a, r, phi_plus1, is_terminal in mini_batch:
            phi_j.append(phi)
            a_j.append(a)
            r_j.append(r)
            phi_j_plus1.append(phi_plus1)
            is_terminal_j.append(is_terminal)
        phi_j = torch.tensor(np.array(phi_j, dtype=np.float32), device=device)
        a_j = torch.tensor(np.array(a_j), device=device)
        r_j = torch.tensor(np.array(r_j, dtype=np.float32), device=device)
        phi_j_plus1 = torch.tensor(np.array(phi_j_plus1, dtype=np.float32), device=device)
        is_terminal_j = torch.tensor(np.array(is_terminal_j), device=device)
        y_j = r_j + torch.logical_not(is_terminal_j) * GAMMA * torch.max(model.forward(phi_j_plus1), axis=1).values

        # Train the network
        out_j = model(phi_j).gather(1, a_j.unsqueeze(-1)).squeeze(-1)
        loss_j = loss_function(out_j, y_j)
        loss_j.backward()
        optimizer.step()
        optimizer.zero_grad()

        phi_t = phi_t_plus1
        n_frames += 1

        if n_frames % 10000 == 0:
            print(dt.now(), f"Done {n_frames} training steps")
            eval = True


    if eval:
        print(dt.now(), "Evaluating performance...")
        episode_rewards = []
        for _ in range(100):
            env.reset()
            terminated = truncated = done = False
            rewards = []
            phi_t = np.zeros((4, 84, 84), dtype=np.float32)
            obs_2_max = np.zeros((2, 84, 84), dtype=np.float32)
            while not done:
                if np.random.uniform() < 0.05:
                    a_t = env.action_space.sample()
                else:
                    action_values = model.forward(torch.from_numpy(phi_t).to(device))
                    a_t = torch.argmax(action_values).item()
                r_t = 0
                for i in range(4):
                    obs, r, terminated, truncated, info = env.step(a_t)
                    done = terminated or truncated or info['lives'] < 5
                    if i >= 2:
                        obs_2_max[i % 2] = preprocess_obs(obs)
                    r_t += r
                    if done: break
                rewards.append(r_t)

                # Store the state transition, action and reward into the replay buffer
                x_t_plus1 = obs_2_max.max(axis=0)
                phi_t_plus1 = np.roll(phi_t, shift=-1, axis=0)
                phi_t_plus1[-1] = x_t_plus1
                phi_t = phi_t_plus1
            episode_rewards.append(sum(rewards))
        print(dt.now(), f'Average reward per episode: {sum(episode_rewards) / len(episode_rewards)}')
        eval = False









