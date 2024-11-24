import random
from collections import defaultdict
import numpy as np
from common import map_state_to_discrete_state, map_discrete_state_to_state, get_next_state
from tqdm import tqdm
import gym


def epsilon_greedy_policy(state, Q, epsilon=0.5):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


def generate_episode(env, num_timesteps, Q, states, epsilon=0.5):
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):
        state = map_state_to_discrete_state(state, states)

        # select the action according to the epsilon-greedy policy
        action = epsilon_greedy_policy(state, Q, epsilon)

        # perform the selected action and store the next state information
        next_state, reward, terminated, truncated, info = env.step(action)
        # next_state = get_next_state(map_discrete_state_to_state(state, states), action)

        # modify the reward, since -1 is always given, we want to give a reward based on the position
        reward = -1 / (np.abs(next_state[0]) * np.abs(next_state[1]) + 1)

        done = terminated  # or truncated

        # store the state, action, reward in the episode list
        episode.append((state, action, reward))

        # if the next state is a final state then break the loop else update the next state to the current
        # state
        if done:
            # print("done")
            break

        state = next_state

    return episode


def generate_policy(Q):
    policy = defaultdict(int)

    states = {state for (state, action) in Q.keys()}
    for state in states:
        policy[state] = max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])
    return policy


def test_policy(policy, env, states):
    num_episodes = 100
    num_timesteps = 1000
    total_reward = 0

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(num_timesteps):
            state = map_state_to_discrete_state(state, states)

            env.render()

            action = policy[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated  # or truncated
            episode_reward += reward

            if done:
                break

            state = next_state

        total_reward += episode_reward

    return total_reward / num_episodes


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')  # , render_mode='human')

    position_bins = 50
    velocity_bins = 20

    states = np.zeros((position_bins, velocity_bins))

    Q = defaultdict(float)
    total_return = defaultdict(float)

    N = defaultdict(int)

    epsilon = 0.7
    epsilon_decay = 0.999

    num_iterations = 1000  # 60000
    for i in tqdm(range(num_iterations)):
        episode = generate_episode(env, 10000, Q, states, epsilon)
        epsilon *= epsilon_decay

        # print(len(episode))
        # get all the state-action pairs in the episode
        all_state_action_pairs = [(s, a) for (s, a, r) in episode]

        # store all the rewards obtained in the episode in the rewards list
        rewards = [r for (s, a, r) in episode]

        # for each step in the episode
        for t, (state, action, reward) in enumerate(episode):

            # if the state-action pair is occurring for the first time in the episode
            if not (state, action) in all_state_action_pairs[0:t]:
                # compute the return R of the state-action pair as the sum of rewards
                R = sum(rewards[t:])

                # update total return of the state-action pair
                total_return[(state, action)] = total_return[(state, action)] + R

                # update the number of times the state-action pair is visited
                N[(state, action)] += 1

                # compute the Q value by just taking the average
                Q[(state, action)] = total_return[(state, action)] / N[(state, action)]

    policy = generate_policy(Q)
    print(policy)

    env = gym.make('MountainCar-v0')  # , render_mode='human')
    print(test_policy(policy, env, states))