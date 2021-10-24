import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class CliffWalk:
    eps = 0.1
    def __init__(self):
        self.eps = 0.1
        self.runs = 100
        self.alpha = 0.2
        self.num_episodes = 500
        self.q_values = {}
        self.initializeqvalues()

    def epsilon_greedy(self, state):
        feasable_actions = self.get_feasable_actions(state)
        u = np.argmax([self.q_values[(state, i)] for i in feasable_actions])
        prob_vector = [self.eps/len(feasable_actions) for _ in range(len(feasable_actions))]
        prob_vector[u] = 1 - self.eps + self.eps/len(feasable_actions)
        return np.random.choice(feasable_actions, p = prob_vector)
    
    def maxQ(self, state):
        feas_act = self.get_feasable_actions(state)
        return np.max([self.q_values[(state, i)] for i in feas_act])

    def get_feasable_actions(self, state):
        x, y = state
        possibleActions = []
        if(x+1 <= 12):
            possibleActions.append('right')
        if(x-1 >= 1):
            possibleActions.append('left')
        if(y+1 <= 3):
            possibleActions.append('up')
        if(y-1 >= 1):
            possibleActions.append('down')
        return possibleActions

    def initializeqvalues(self):
        for i in range(1, 13):
            for j in range(1, 5):
                state = (i, j)
                for act in self.get_feasable_actions(state):
                    self.q_values[(state, act)] = 0
    
    def getNextState(self, current_state, action):
        x, y = current_state
        reward = -1
        terminal = False
        if action == "up":
            next_state =  (x, y+1)
        if action == "down":
            next_state = (x, y-1)
        if action == "right":
            next_state = (x+1, y)
        if action =="left":
            next_state = (x-1, y)
        
        xnew , ynew = next_state
        if 1 < xnew < 12 and ynew == 1:
            next_state = (1,1)
            reward = -100
        
        if next_state == (12, 1):
            terminal = True
        
        return next_state, reward, terminal

    def qlearning(self):
        reward_sum = [0 for _ in range(self.num_episodes)]
        for _ in tqdm(range(self.runs), "Q-Learning"):
            self.initializeqvalues()
            for ep in range(self.num_episodes):
                state = (1,1)
                terminal = False
                while not terminal:
                    action = self.epsilon_greedy(state)
                    next_state, reward, terminal = self.getNextState(state, action)
                    reward_sum[ep] += reward
                    self.q_values[(state, action)] += self.alpha*(reward + self.maxQ(next_state) - self.q_values[(state, action)])
                    state = next_state
        return np.array(reward_sum)/self.runs

    def sarsa(self):
        reward_sum = [0 for _ in range(self.num_episodes)]
        for _ in tqdm(range(self.runs), "SARSA"):
            self.initializeqvalues()
            for ep in range(self.num_episodes):
                state = (1,1)
                action = self.epsilon_greedy(state)
                terminal = False
                while not terminal:
                    next_state, reward, terminal = self.getNextState(state, action)
                    reward_sum[ep] += reward
                    next_action = self.epsilon_greedy(next_state)
                    self.q_values[(state, action)] += self.alpha*(reward + self.q_values[(next_state, next_action)] - self.q_values[(state, action)])
                    state = next_state
                    action = next_action
        return np.array(reward_sum)/self.runs

c = CliffWalk()
sum_rewards_q_learning = c.qlearning()
sum_rewards_sarsa = c.sarsa()

plt.plot(list(range(500)), sum_rewards_q_learning, label = "Q - Learning")
plt.plot(list(range(500)), sum_rewards_sarsa, label = "SARSA")
plt.legend()
plt.ylabel("Emprirical Sun of Rewards over Runs")
plt.xlabel("Episodes")
plt.ylim(-150, 0)
plt.show()