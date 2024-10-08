from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical

torch.manual_seed(0)  # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=128, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(env, policy, optimizer, scheduler, n_episodes=2000, max_t=1000, gamma=0.99, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)

            # Penalize the cart for being far from the center
            position_penalty = -abs(state[0]) / 4.8
            reward += position_penalty

            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        scheduler.step()

        if i_episode % print_every == 0:
            torch.save(policy.state_dict(), 'checkpoint.pth')
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 475.0:
                print(f'Solved in {i_episode} episodes!')
                break
    return scores

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    policy = Policy(s_size=4, a_size=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.0007)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    scores = reinforce(env, policy=policy, optimizer=optimizer, scheduler=scheduler, gamma=0.99)