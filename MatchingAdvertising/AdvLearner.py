from Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern as M


class AdvLearner(Learner):
    def __init__(self, n_arms,n_ads,n_bids, n_budget,t):
        super().__init__(n_arms)
        self.arms = np.array([[np.zeros(shape=1) for j in range(n_ads)] for i in range(n_ads)])
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        alpha = 10.0
        #kernel =C(1.0, (1e-3, 1e3)) * RBF([2111111111111, 1], (1e-2, 1e2))
        #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha **2, n_restarts_optimizer=10)
        self.input = np.array([[5000, 25],[5000, 50],[5000, 75],[5000, 100],[10000, 25],[10000, 50],[10000, 75],[10000, 100],[20000, 25],[20000, 50],[20000, 75],[20000, 100],[30000, 25],[30000, 50],[30000, 75],[30000, 100]])
        self.rewards_per_arm = [[[] for j in range(n_bids)] for i in range(n_budget)]
        self.collected_rewards = [[] for j in range(t)]
        self.collected_rewardsy = np.array([])


    def update(self,arm, reward,t):
        self.t += 1
        self.update_observation(arm, reward,t)
        self.update_model(t)

    def update_observation(self, arm_idx, reward,t):
        #self.rewards_per_arm[arm_.append(reward)]
        self.collected_rewardsy = np.append(self.collected_rewardsy, reward)
        #self.collected_rewards[t].append(reward)
        value = np.array([(arm_idx[0]+1) *5000.0, (arm_idx[1]+1) *25.0])
        self.pulled_arms.append(value)
        # [3,1]
    def update_model(self,t):
        x =np.atleast_2d(self.pulled_arms)
        #X = X.T
        #x = np.atleast_2d(self.pulled_arms).T
        # y = []
        # for i in range(t):
        #     for j in range(len(self.collected_rewards[t])+1):
        #         y.append(self.collected_rewards[i][j])
        y = self.collected_rewardsy #N click
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.input), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def estimate_n(self):
        #print(np.random.normal(self.means,self.sigmas))
        return np.random.normal(self.means,self.sigmas)
    def update_reward(self,reward,t):
        self.collected_rewards[t].append(reward)




