import gym
import numpy as np
import time

def show_random_agent(env):
    state = env.reset()
    for t in range(1000):
        env.render()
        state, reward, done, _ = env.step(np.random.randint(0, 4))
        if done:
            break
        time.sleep(0.05)
    env.close()

def compute_value_function(policy, env, num_iterations=1000, threshold=1e-20, gamma=1.0):
    value_table = np.zeros(env.observation_space.n)
    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            a = policy[s]
            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
                                  for prob, s_, r, _ in env.P[s][a]])
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table

def extract_policy(value_table, env, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        Q_values = [sum([prob * (r + gamma * value_table[s_])
                         for prob, s_, r, _ in env.P[s][a]])
                    for a in range(env.action_space.n)]
        policy[s] = np.argmax(np.array(Q_values))
    return policy

def policy_iteration(env):
    num_iterations = 1000
    policy = np.zeros(env.observation_space.n)
    for i in range(num_iterations):
        value_function = compute_value_function(policy, env)
        new_policy = extract_policy(value_function, env)
        if (np.all(policy == new_policy)):
            break
        policy = new_policy
    return policy

def test(env, optimal_policy, render=True):
    state, _ = env.reset()
    if render:
        env.render()
    total_reward = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, _, _ = env.step(action)
        if render:
            env.render()
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
    optimal_policy = policy_iteration(env)
    print(optimal_policy)
    total_reward = test(env, optimal_policy)
    print(total_reward)
    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, optimal_policy, render=False)
        sum_reward += total_reward
    print(sum_reward / 5000)
    env.close()