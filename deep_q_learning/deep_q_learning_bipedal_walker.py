import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, s_size=24, h_size=128, a_size=4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, h_size // 2)
        self.fc4 = nn.Linear(h_size // 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))

    def act(self, state, noise, epsilon=0.0):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self(state).cpu().numpy()
        action += epsilon * noise()
        return np.clip(action, -1, 1)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = Policy(s_size=state_dim, a_size=action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * torch.max(
                self.model(torch.tensor(next_state, dtype=torch.float32, device=device))).item()

            with torch.no_grad():
                target_f = self.model(torch.tensor(state, dtype=torch.float32, device=device)).cpu().numpy()
                target_f = torch.tensor(target_f, dtype=torch.float32, device=device)
                target_f[np.argmax(action)] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.tensor(state, dtype=torch.float32, device=device)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


def test(env, policy, render=True, num_episodes=1):
    total_reward = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            state = torch.tensor(state[0], dtype=torch.float32, device=device)  # env.reset() returns a tuple (obs, info)
            for _ in range(1000):
                action = policy.act(state, lambda: 0)  # No noise during testing
                next_state, reward, done, info, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if render:
                    env.render()
                    time.sleep(0.05)

                total_reward += reward
                if done:
                    break
                state = next_state

    print(f'Total Reward: {total_reward / num_episodes}')

    return total_reward / num_episodes


def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DQNAgent(state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=20000)

    scores = []
    scores_deque = deque(maxlen=100)

    best_reward = -np.inf

    for i_episode in range(200):
        state = env.reset()
        state = state[0]  # env.reset() now returns a tuple (obs, info)
        score = 0
        for t in range(1000):
            action = agent.model.act(state, noise=agent.noise, epsilon=agent.epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            reward /= 10.0  # Scale reward
            agent.remember(state, action, reward, next_state, done or truncated)

            state = next_state
            score += reward

            agent.replay(64)

            if done or truncated:
                break

        scores_deque.append(score)
        scores.append(score)
        print(f'Episode {i_episode} Score {score} Average Score {np.mean(scores_deque)}')

        if i_episode % 30 == 29:
            reward = test(env, agent.model, render=False, num_episodes=5)

            if reward > best_reward:
                best_reward = reward
                torch.save(agent.model.state_dict(), 'checkpoint_best.pth')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    # train(env)

    policy = Policy(s_size=24, a_size=4).to(device)
    policy.load_state_dict(torch.load('checkpoint_best.pth'))

    env = gym.make('BipedalWalker-v3', render_mode='human')

    test(env, policy, render=True, num_episodes=30)