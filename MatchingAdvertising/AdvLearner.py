from Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class AdvLearner(Learner):
    def __init__(self, n_arms, arms,n_bids, n_budget,t):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha **2, normalize_y=True, n_restarts_optimizer=10)

        self.rewards_per_arm = [[[] for j in range(n_bids)] for i in range(n_budget)]
        self.collected_rewards = [[] for j in range(t)]


    def update(self,superarm, reward):
        self.t += 1
        for arm_i, arm in enumerate(superarm, start=0):
            self.update_observation(arm, reward[arm_i])
        self.update_model()

    def update_observation(self, arm, reward):
        self.rewards_per_arm[arm[0]][arm[1]].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(self.arms[arm[0]][arm[1]])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def estimate_n(self):
        return np.random.normal(self.means,self.sigmas)




