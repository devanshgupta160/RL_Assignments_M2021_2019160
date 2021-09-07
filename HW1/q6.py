import numpy as np
import matplotlib.pyplot as plt
from testbed import MultiArmedBandit

def c2(iter_no):
    return 0.1

num_total_runs = 2000
total_iter = 1000
arms = 10

cumu_rewards = {"UCB":{}}

mab = MultiArmedBandit(arms)
rewards_obtained, _, _ = mab.run(total_iter, num_total_runs, c2)
cumu_rewards["Epsilon Greedy Ɛ = 0.01"] = rewards_obtained

mab = MultiArmedBandit(arms, UCB = True)
rewards_obtained, _, _ = mab.run(total_iter, num_total_runs, c2, c = 1)
cumu_rewards["UCB"]["UCB c=1"] = rewards_obtained

rewards_obtained, _, _ = mab.run(total_iter, num_total_runs, c2, c = 2)
cumu_rewards["UCB"]["UCB c=2"] = rewards_obtained

rewards_obtained, _, _ = mab.run(total_iter, num_total_runs, c2, c = 4)
cumu_rewards["UCB"]["UCB c=4"] = rewards_obtained

x = np.arange(1, total_iter+1)
for k in cumu_rewards["UCB"].keys():
    plt.plot(x, cumu_rewards["Epsilon Greedy Ɛ = 0.01"], label = "Epsilon Greedy Ɛ = 0.01")
    plt.plot(x, cumu_rewards["UCB"][k], label = k)
    plt.ylabel("Average Reward")
    plt.xlabel("Iterations")
    plt.title("Average Reward obtained over Iterations")
    plt.legend()
    plt.savefig(f"q6/{k}_vs_Epsilon_Greedy.jpg")
    plt.cla()