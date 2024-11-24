# SARSA

## Setup
```python
# Parameters
alpha = 0.01
gamma = 0.95
epsilon = 0.7

num_episodes = 500000
num_timesteps = 5000

# Epsilon decay
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
```

## Results
```markdown
defaultdict(<class 'int'>, {0: 0, 1: 3, 2: 0, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 3, 9: 1, 10: 0, 11: 0, 12: 0, 13: 2, 14: 1, 15: 0})
0.78078
```