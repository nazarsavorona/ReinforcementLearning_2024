import gym
import numpy as np
from tqdm import tqdm
from common import map_state_to_discrete_state, map_discrete_state_to_state, get_next_state, test


def compute_value_function(policy, num_iterations=10, threshold=1e-4, gamma=1.0):
    value_table = np.zeros(policy.shape)

    for i in tqdm(range(num_iterations)):
        updated_value_table = np.copy(value_table)
        for pos_idx in range(states.shape[0]):
            for vel_idx in range(states.shape[1]):
                s_idxs = (pos_idx, vel_idx)
                s = map_discrete_state_to_state(s_idxs, states)
                a = int(policy[s_idxs])
                next_state = get_next_state(s, a)
                next_state_idxs = map_state_to_discrete_state(next_state, states)
                reward = -1 + np.abs(next_state[0] - s[0])
                value_table[s_idxs] = reward + gamma * updated_value_table[next_state_idxs]

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table


def extract_policy(value_table, env, gamma=1.0):
    policy = np.zeros((states.shape[0], states.shape[1]))

    for pos_idx in range(states.shape[0]):
        for vel_idx in range(states.shape[1]):
            states_idxs = (pos_idx, vel_idx)
            s = map_discrete_state_to_state(states_idxs, states)
            q_values = []
            for a in range(env.action_space.n):
                next_state = get_next_state(s, a)
                next_state_idxs = map_state_to_discrete_state(next_state, states)
                reward = -1 + np.abs(next_state[0] - s[0])
                q_values.append(reward + gamma * value_table[next_state_idxs])
            policy[states_idxs] = np.argmax(np.array(q_values))
    return policy


def policy_iteration(env, states, num_iterations=10):
    policy = np.zeros((states.shape[0], states.shape[1]))
    for i in tqdm(range(num_iterations)):
        value_function = compute_value_function(policy)
        new_policy = extract_policy(value_function, env)
        if (np.all(policy == new_policy)):
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='human')

    # Create discrete state space
    position_bins = 20
    velocity_bins = 20

    states = np.zeros((position_bins, velocity_bins))

    optimal_policy = policy_iteration(env, states, num_iterations=10)
    print(optimal_policy)

    total_reward = test(env, optimal_policy)
    print(total_reward)
    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, optimal_policy, render=False)
        sum_reward += total_reward
    print(sum_reward / 5000)
    env.close()
