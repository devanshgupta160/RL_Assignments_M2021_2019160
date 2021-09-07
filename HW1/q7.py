import matplotlib.pyplot as plt
import numpy as np

def softmax(H):
    return np.exp(H)/np.sum(np.exp(H))

class GradientBandits:
    def __init__(self, n, mu_ = 0, var_ = 1, var = 1, baseline = True):
        self.n = n
        self.mu_ = mu_
        self.var_ = var_
        self.var = var
        self.H_t = np.zeros(self.n,)
        self.pi_t = None
        self.baseline = baseline
        self.reward = None
    
    def initialize(self):
        self.H_t = np.zeros(self.n,)
        self.pi_t = None
        self.reward = None

    def select_action(self):
        self.pi_t = softmax(self.H_t)
        return np.random.choice(list(range(self.n)), p = self.pi_t)
    
    def get_rewards(self, mus):
        self.reward = self.var**0.5 * np.random.randn(self.n,) + mus
    
    def update(self, a, reward, baseline, learning_rate = 0.01):
        self.H_t[a] = self.H_t[a] + learning_rate*(reward - baseline)*(1 - self.pi_t[a])
        for i in range(self.n):
            if i != a:
                self.H_t[i] = self.H_t[i] - learning_rate*(reward - baseline)*self.pi_t[i]

    def run(self, total_iter, tot_runs, learning_rate = 0.01):
        self.initialize()
        optimal_action_taken = []
        for _ in range(tot_runs):
            reward = 0
            mus = self.var_**0.5 * np.random.randn(self.n,) + self.mu_
            op_ = []
            self.initialize()
            for it in range(total_iter):
                a = self.select_action()
                op_.append(a==np.argmax(mus))
                self.get_rewards(mus)
                reward += self.reward[a]
                if self.baseline:
                    self.update(a, self.reward[a], reward/(it+1), learning_rate)
                else:
                    self.update(a, self.reward[a], 0, learning_rate)
            optimal_action_taken.append(np.array(op_))
        return np.mean(np.array(optimal_action_taken), axis = 0)*100                

total_iter = 1000
total_runs = 2000
arms = 10

optim = {}

mab = GradientBandits(arms, mu_=4, baseline=True)
optim["Baseline α = 0.1"] = mab.run(total_iter, total_runs, 0.1)
optim["Baseline α = 0.4"] = mab.run(total_iter, total_runs, 0.4)

mab = GradientBandits(arms, mu_=4, baseline=False)
optim["No Baseline α = 0.1"] = mab.run(total_iter, total_runs, 0.1)
optim["No Baseline α = 0.4"] = mab.run(total_iter, total_runs, 0.4)

x = np.arange(1, total_iter+1)
for k in optim.keys():
    plt.plot(x, optim[k], label = k)
plt.ylabel("Percentage for Optimal Actions")
plt.xlabel("Iterations")
plt.title("Percentage time optimal action taken over Iterations")
plt.legend()
plt.savefig("optimal_action_gradient_bandits.jpg")
plt.cla()
