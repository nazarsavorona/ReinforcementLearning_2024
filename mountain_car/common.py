import numpy as np


# Map state to discrete state
def map_state_to_discrete_state(state, states):
    position, velocity = state

    linspace_pos = np.linspace(-1.2, 0.6, len(states[0]))
    linspace_vel = np.linspace(-0.07, 0.07, len(states[1]))

    position_idx = np.digitize(position, linspace_pos) - 1
    velocity_idx = np.digitize(velocity, linspace_vel) - 1

    # Ensure the indices are within bounds
    position_idx = max(0, min(position_idx, len(states[0]) - 1))
    velocity_idx = max(0, min(velocity_idx, len(states[1]) - 1))

    return position_idx, velocity_idx


# Map discrete state to state
def map_discrete_state_to_state(discrete_state, states):
    position_idx, velocity_idx = discrete_state

    pos_linspace = np.linspace(-1.2, 0.6, len(states[0]))
    vel_linspace = np.linspace(-0.07, 0.07, len(states[1]))

    # Ensure the indices are within bounds
    position_idx = max(0, min(position_idx, len(states[0]) - 1))
    velocity_idx = max(0, min(velocity_idx, len(states[1]) - 1))

    position = pos_linspace[position_idx]
    velocity = vel_linspace[velocity_idx]

    return position, velocity


def get_next_state(state, action):
    force = 0.001
    gravity = 0.0025

    position, velocity = state

    velocity_next = velocity + (action - 1) * force - np.cos(3 * position) * gravity
    velocity_next = np.clip(velocity_next, -0.07, 0.07)

    position_next = position + velocity_next
    position_next = np.clip(position_next, -1.2, 0.6)

    return position_next, velocity_next


def test(env, optimal_policy, render=True):
    state, _ = env.reset()
    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):
        state_idxs = map_state_to_discrete_state(state, optimal_policy)

        action = int(optimal_policy[state_idxs])
        state, reward, done, info, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward
