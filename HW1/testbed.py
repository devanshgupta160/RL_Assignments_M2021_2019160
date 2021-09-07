import numpy as np
import random

class MultiArmedBandit:
    def __init__(self, n, mu_ = 0, var_ = 1, var = 1, UCB = False, stationarity = True):
        self.n = n
        self.mu_ = mu_
        self.var_ = var_
        self.var = var
        self.UCB = UCB
        self.stationarity = stationarity
        self.Q_t = np.zeros(self.n,)
        self.reward = None
    
    def initialize(self):
        self.Q_t = np.zeros(self.n,)
        self.reward = None
    
    def select_arm(self, eps = 0, iter_ = 0, c = 1, tracker = None):
        if self.UCB:
            return np.argmax(self.Q_t + c*(np.log(iter_)/(tracker + 1e-20))**0.5)
        a = np.random.choice([0, 1], p = [1 - eps, eps])
        if a == 0:
            return np.argmax(self.Q_t)
        else:
            return random.choice(range(self.n))
    
    def get_rewards(self, mus):
        self.reward = self.var**0.5 * np.random.randn(self.n,) + mus
    
    def update_q(self, a, step):
        self.Q_t[a] = self.Q_t[a] + step * (self.reward[a] - self.Q_t[a])
    
    def run(self, total_iter, tot_runs, eps, step = None, c = None):
        reward_obtained_f = []
        optimal_actions_selected_f = []
        abs_error_f = []
        for _ in range(tot_runs):
            tracker = np.zeros(self.n,)
            reward_obtained = []
            optimal_actions_selected = []
            abs_error = []
            self.initialize()
            if self.stationarity:
                mus = self.var_**0.5 * np.random.randn(self.n,) + self.mu_
            else:
                mus = np.random.randn() * np.ones(self.n,)
            for iter_no in range(total_iter):
                if self.UCB:
                    a = self.select_arm(eps(iter_no+1), iter_ = iter_no+1, c=c, tracker=tracker)
                else:
                    a = self.select_arm(eps(iter_no+1))
                tracker[a] += 1
                self.get_rewards(mus)
                if not self.stationarity:
                    mus += 0.01*np.random.randn(self.n,)
                optimal_actions_selected.append(a==np.argmax(mus))
                reward_obtained.append(self.reward[a])
                abs_error.append(np.abs(self.Q_t - mus))
                if step:
                    self.update_q(a, step)
                else:
                    self.update_q(a, 1/tracker[a])
            reward_obtained_f.append(np.array(reward_obtained))
            optimal_actions_selected_f.append(np.array(optimal_actions_selected))
            abs_error_f.append(np.array(abs_error))
        return np.mean(np.array(reward_obtained_f), axis = 0), np.mean(np.array(optimal_actions_selected_f), axis = 0)*100, np.mean(np.array(abs_error_f), axis = 0)
