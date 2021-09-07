import numpy as np
import matplotlib.pyplot as plt
from testbed import MultiArmedBandit

def c2(iter_no):
    return 0.1

num_total_runs = 2000
total_iter = 10000
arms = 10

cumu_rewards = {}
optim = {}

mab = MultiArmedBandit(arms, stationarity=False)
rewards_obtained, optim_, _ = mab.run(total_iter, num_total_runs, c2)
cumu_rewards["Sample Mean"] = rewards_obtained
optim["Sample Mean"] = optim_

rewards_obtained, optim_, _ = mab.run(total_iter, num_total_runs, c2, step=0.1)
cumu_rewards["α = 0.1"] = rewards_obtained
optim["α = 0.1"] = optim_

x = np.arange(1, total_iter+1)
for k in cumu_rewards.keys():
    plt.plot(x, cumu_rewards[k], label = k)
plt.ylabel("Average Reward")
plt.xlabel("Iterations")
plt.title("Average Reward obtained over Iterations in a Non-Stationary setting")
plt.legend()
plt.savefig("images/non_stationary_reward.jpg")
plt.cla()

x = np.arange(1, total_iter+1)
for k in cumu_rewards.keys():
    plt.plot(x, optim[k], label = k)
plt.ylabel("Percentage of Optimal Actions")
plt.xlabel("Iterations")
plt.title("Percentage of Optimal Actions taken over Iterations in a Non-Stationary setting")
plt.legend()
plt.savefig("images/non_stationary_optimal.jpg")