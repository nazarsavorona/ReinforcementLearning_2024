import gym
import random

from collections import defaultdict

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

def generate_policy(Q):

    policy = defaultdict(int)

    for state in range(env.observation_space.n):
        policy[state] = max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])

    return policy

def sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon):
    Q = defaultdict(float)

    for i in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        for t in range(num_timesteps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state
            action = next_action

            if done:
                break
        show_Q(env, Q)

    return Q

def test(env, optimal_policy, render=True):

    state = env.reset()
    if render:
        env.render()


    total_reward = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, info = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward

def show_agent(env, policy):
    state = env.reset()
    env.render()

    for t in range(1000):

        state, reward, done, _ = env.step(policy[state])
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
        print(table.reshape(4,4))



if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    env.seed(42)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9

    num_episodes = 5000
    num_timesteps = 1000

    Q = sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon)

    policy = generate_policy(Q)

    print(policy)

    show_agent(env, policy)

    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, policy, render=False)
        sum_reward += total_reward

    print(sum_reward / 5000)

    env.close()

