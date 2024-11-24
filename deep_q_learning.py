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
    def __init__(self, s_size=6, h_size=32, a_size=3):
        super(Policy, self).__init__()
        self.a_size = a_size

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.a_size)
        state = state.to(device)  # Ensure state is on the correct device
        q_values = self(state)
        return torch.argmax(q_values).item()


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(
                    self.model(torch.tensor(next_state, dtype=torch.float32, device=device))).item()

            with torch.no_grad():
                target_f = self.model(torch.tensor(state, dtype=torch.float32, device=device)).numpy()
                target_f[action] = target
                target_f = torch.tensor(target_f, dtype=torch.float32, device=device)

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.tensor(state, dtype=torch.float32, device=device)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


def test(env, policy, render=True, num_episodes=1):
    if render:
        env.render()

    total_reward = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)  # Ensure state is on the correct device
            for _ in range(1000):
                action = policy.act(state)
                next_state, reward, done, info, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32,
                                          device=device)  # Ensure next_state is on the correct device

                if render:
                    env.render()
                    time.sleep(0.05)

                total_reward += reward
                if done:
                    break
                state = next_state  # Update state

    print(f'Total Reward: {total_reward / num_episodes}')

    return total_reward / num_episodes


def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=1000)

    scores = []
    scores_deque = deque(maxlen=100)

    best_reward = -np.inf

    for i_episode in range(200):
        state, _ = env.reset()
        score = 0
        for t in range(1000):
            action = agent.model.act(state, epsilon=agent.epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.replay(16)

            if done:
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
    env = gym.make('Acrobot-v1')
    # env.seed(0)

    train(env)

    policy = Policy(s_size=6, a_size=3).to(device)
    policy.load_state_dict(torch.load('checkpoint_best.pth'))

    test(env, policy, render=False, num_episodes=30)
    test(env, policy, render=True)
