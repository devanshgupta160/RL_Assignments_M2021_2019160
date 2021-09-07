import numpy as np
import matplotlib.pyplot as plt
from testbed import MultiArmedBandit

def invert(iter_no):
    return 1/iter_no

def c1(iter_no):
    return 0.01

def c2(iter_no):
    return 0.1

def greedy(iter_no):
    return 0


num_total_runs = 2000
total_iter = 1000
arms = 10

cumu_rewards = {}
optim_act = {}
abs_error = {}

mab = MultiArmedBandit(arms)
rewards_obtained, optimal_actions_selected, a_e = mab.run(total_iter, num_total_runs, greedy)
cumu_rewards["Greedy"] = rewards_obtained
optim_act["Greedy"] = optimal_actions_selected
abs_error["Greedy"] = a_e

rewards_obtained, optimal_actions_selected, a_e = mab.run(total_iter, num_total_runs, c1)
cumu_rewards["Ɛ = 0.01"] = rewards_obtained
optim_act["Ɛ = 0.01"] = optimal_actions_selected
abs_error["Ɛ = 0.01"] = a_e

rewards_obtained, optimal_actions_selected, a_e = mab.run(total_iter, num_total_runs, c2)
cumu_rewards["Ɛ = 0.1"] = rewards_obtained
optim_act["Ɛ = 0.1"] = optimal_actions_selected
abs_error["Ɛ = 0.1"] = a_e

rewards_obtained, optimal_actions_selected, a_e = mab.run(total_iter, num_total_runs, invert)
cumu_rewards["Ɛ(t) = 1/t"] = rewards_obtained
optim_act["Ɛ(t) = 1/t"] = optimal_actions_selected
abs_error["Ɛ(t) = 1/t"] = a_e

x = np.arange(1, total_iter+1)

#Plotting the average reward obtained and the percentage optimal actions taken plots
for k in cumu_rewards.keys():
    plt.plot(x, cumu_rewards[k], label = k)
plt.ylabel("Average Reward")
plt.xlabel("Iterations")
plt.title("Average Reward obtained over Iterations")
plt.legend()
plt.savefig("q1/reward.jpg")
plt.cla()

for k in optim_act.keys():
    plt.plot(x, optim_act[k], label = k)
plt.ylabel("Percentage for Optimal Actions")
plt.xlabel("Iterations")
plt.title("Percentage time optimal action taken over Iterations")
plt.legend()
plt.savefig("q1/optimal_action.jpg")
plt.cla()

for arm in range(arms):
    for k in abs_error.keys():
        plt.plot(x, abs_error[k][:,arm], label = k)
    plt.ylabel(f"Mean Absolute Error for arm {arm}")
    plt.xlabel("Iterations")
    plt.title(f"Mean Absolute Error for arm {arm} over Iterations")
    plt.legend()
    plt.savefig(f"q1/mean_absolute_error_{arm}.jpg")
    plt.cla()