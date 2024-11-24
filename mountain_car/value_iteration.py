import gym
import numpy as np

from common import map_state_to_discrete_state, map_discrete_state_to_state, get_next_state, test


# Value iteration method
def value_iteration(env, states, num_iterations=1000, threshold=1e-20, gamma=0.99):
    # Initialize value table to zeros
    value_table = np.zeros(states.shape)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for pos_idx in range(states.shape[0]):
            for vel_idx in range(states.shape[1]):
                s_idxs = (pos_idx, vel_idx)
                s = map_discrete_state_to_state(s_idxs, states)

                q_values = []
                for a in range(env.action_space.n):
                    next_state = get_next_state(s, a)
                    next_state_idxs = map_state_to_discrete_state(next_state, states)

                    reward = -1 + np.abs(next_state[0] - s[0])

                    # Bellman update: Q-value + discounted future reward
                    q_value = reward + gamma * updated_value_table[next_state_idxs]
                    q_values.append(q_value)

                # Take the maximum over the actions for value iteration
                value_table[s_idxs] = max(q_values)

        # Check for convergence
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table


def extract_policy(env, value_table, states):
    gamma = 1.0
    policy = np.zeros((states.shape[0], states.shape[1]))

    for pos_idx in range(states.shape[0]):
        for vel_idx in range(states.shape[1]):
            s_idxs = (pos_idx, vel_idx)
            s = map_discrete_state_to_state(s_idxs, states)

            q_values = []
            for a in range(env.action_space.n):
                next_state = get_next_state(s, a)
                next_state_idxs = map_state_to_discrete_state(next_state, states)

                reward = -1 + np.abs(next_state[0] - s[0])

                q_values.append(reward + gamma * value_table[next_state_idxs])

            # Choose the action with the highest Q-value
            policy[s_idxs] = np.argmax(q_values)

    return policy


if __name__ == '__main__':

    env = gym.make('MountainCar-v0', render_mode='rgb_array')

    # Create discrete state space
    position_bins = 50
    velocity_bins = 30
    states = np.zeros((position_bins, velocity_bins))

    optimal_value_function = value_iteration(env=env, states=states, num_iterations=100)

    optimal_policy = extract_policy(env, optimal_value_function, states)

    print(optimal_policy)

    total_reward = test(env, optimal_policy, states)
    print(total_reward)

    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, optimal_policy, render=False)
        sum_reward += total_reward

    print(sum_reward / 5000)

    env.close()
