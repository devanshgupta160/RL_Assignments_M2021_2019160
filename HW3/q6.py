import random
import matplotlib.pyplot as plt
import numpy as np
import copy

gt = np.array([1/6, 2/6, 3/6, 4/6, 5/6], dtype = np.float)

def get_sequence():
    states = []
    rewards = []
    currentState = 2
    while currentState != -1 and currentState != 5:
        states.append(currentState)
        currentState = currentState + random.choice([1, -1])
        if currentState == 5:
            reward = 1
        else:
            reward = 0
        rewards.append(reward)
    return states, rewards

def td0_fig1(num_episodes, alpha, state_value):
    for _ in range(num_episodes):
        states, rewards = get_sequence()
        for i , st in enumerate(states):
            next_state = states[i+1] if i < len(states) - 1 else None
            reward = rewards[i]
            if next_state:
                st_ = state_value[next_state]
            else:
                st_ = 0
            state_value[st] = state_value[st] + alpha*(reward + st_ - state_value[st])
    return state_value

def temporal_difference(num_episodes, alpha, num_runs):
    mse_vals = [0 for _ in range(num_episodes)]
    for _ in range(num_runs):
        value = [0.5 for _ in range(5)]
        for ep in range(num_episodes):
            states, rewards = get_sequence()
            for idx, s in enumerate(states):
                r = rewards[idx]
                st_ = 0
                if(idx < len(states) - 1):
                    ns = states[idx+1]
                    st_ = value[ns]
                value[s] += alpha*(r + st_ - value[s])
            mse_vals[ep] += np.mean((np.array(value) - gt)**2)**0.5
    return np.array(mse_vals)/num_runs


def monte_carlo(num_episodes, alpha, num_runs):
    mse_vals = [0 for _ in range(num_episodes)]
    for _ in range(num_runs):
        state_value = [0.5 for _ in range(5)]
        for ep in range(num_episodes):
            states, rewards = get_sequence()
            G = 0
            for i  in range(len(states)-1, -1, -1):
                st_ = states[i]
                rw_ = rewards[i]
                G += rw_
                state_value[st_] += alpha*(G - state_value[st_])
            mse_vals[ep] += np.mean((np.array(state_value) - gt)**2)**0.5
    return np.array(mse_vals)/num_runs

#Figure 1
state_val = np.array([0.5 for _ in range(5)])
plt.plot(['A', 'B', 'C', 'D', 'E'], state_val, label = "Initial")
state_val = td0_fig1(1, 0.1, state_val)
plt.plot(['A', 'B', 'C', 'D', 'E'], state_val, label = "1 Episode")
state_val = td0_fig1(9, 0.1, state_val)
plt.plot(['A', 'B', 'C', 'D', 'E'], state_val, label = "10 Episodes")
state_val = td0_fig1(90, 0.1, state_val)
plt.plot(['A', 'B', 'C', 'D', 'E'], state_val, label = "100 Episodes")
plt.plot(['A', 'B', 'C', 'D', 'E'], gt, label = "True Value")
plt.legend()
plt.xlabel("State")
plt.ylabel("Value")
plt.show()

#Figure 2
mse_td_005 = temporal_difference(100, 0.05 ,100)
mse_td_010 = temporal_difference(100, 0.1 ,100)
mse_td_015 = temporal_difference(100, 0.15 ,100)

mse_mc_001 = monte_carlo(100, 0.01 ,100)
mse_mc_002 = monte_carlo(100, 0.02 ,100)
mse_mc_003 = monte_carlo(100, 0.03 ,100)
mse_mc_004 = monte_carlo(100, 0.04 ,100)

plt.plot(list(range(100)), mse_td_005, label = "TD(alpha = 0.05)")
plt.plot(list(range(100)), mse_td_010, label = "TD(alpha = 0.1)")
plt.plot(list(range(100)), mse_td_015, label = "TD(alpha = 0.15)")
plt.plot(list(range(100)), mse_mc_001, label = "MC(alpha = 0.01)")
plt.plot(list(range(100)), mse_mc_002, label = "MC(alpha = 0.02)")
plt.plot(list(range(100)), mse_mc_003, label = "MC(alpha = 0.03)")
plt.plot(list(range(100)), mse_mc_004, label = "MC(alpha = 0.04)")
plt.legend()
plt.ylabel("Emprirical RMS Error over Runs")
plt.xlabel("Walks/Episodes")
plt.show()