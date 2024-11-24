import gym
import random

from collections import defaultdict

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


def generate_policy(Q):
    policy = defaultdict(int)

    for state in range(env.observation_space.n):
        policy[state] = max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])

    return policy


def sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon):
    Q = defaultdict(float)
    epsilon_decay = 0.995  # Slowly reduce epsilon

    for i in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        for t in range(num_timesteps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state
            action = next_action

            if done:
                break

        # Reduce epsilon after each episode
        epsilon = max(epsilon * epsilon_decay, 0.1)

    return Q


def test(env, optimal_policy, render=True):
    state, _ = env.reset()
    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, info, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward


def show_agent(env, policy):
    state, _ = env.reset()
    env.render()

    for t in range(1000):
        state, reward, done, _, _ = env.step(policy[state])
        env.render()
        if done:
            break

    env.close()


def generate_random_policy(env):
    policy = defaultdict(int)

    for state in range(env.observation_space.n):
        policy[state] = env.action_space.sample()

    return policy


def show_Q(env, Q):
    print("************************************")
    for action in range(env.action_space.n):
        table = np.array([Q[(state, action)] for state in range(env.observation_space.n)])
        print(table.reshape(4, 4))


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    # env.seed(42)

    alpha = 0.01
    gamma = 0.95
    epsilon = 0.7

    num_episodes = 500000
    num_timesteps = 5000

    Q = sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon)

    policy = generate_policy(Q)

    print(policy)

    # show_agent(env, policy)

    sum_reward = 0
    test_number = 50000
    for _ in range(test_number):
        total_reward = test(env, policy, render=False)
        sum_reward += total_reward

    print(sum_reward / test_number)

    env.close()
