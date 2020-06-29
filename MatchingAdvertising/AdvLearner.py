from Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern as M


class AdvLearner(Learner):
    def __init__(self, n_arms,n_ads,n_bids, n_budget,t):
        super().__init__(n_arms)
        self.arms = np.array([[np.ones(shape=1) for j in range(n_ads)] for i in range(n_ads)])
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        alpha = 10.0
        #kernel =C(1.0, (1e-3, 1e3)) * RBF([2111111111111, 1], (1e-2, 1e2))
        #kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha **2, n_restarts_optimizer=10)

        self.rewards_per_arm = [[[] for j in range(n_bids)] for i in range(n_budget)]

    def update(self,arm, reward):
        self.t += 1
        self.update_observation(arm, reward)
        self.update_model()

    def update_observation(self, arm_idx, reward):
        #self.rewards_per_arm[arm_.append(reward)]
        self.collected_rewards = np.append(self.collected_rewards, reward)
        value = np.array([(arm_idx[0]+1) *5000, (arm_idx[1]+1) *25])
        self.pulled_arms.append(value)
        print("pulled_arms", self.pulled_arms)
        # [3,1]
    def update_model(self):
        x =np.atleast_2d(self.pulled_arms).T
        print("x", x)
        #X = X.T
        #x = np.atleast_2d(self.pulled_arms).T
        print(self.collected_rewards)
        y = self.collected_rewards #N click
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def estimate_n(self):
        return np.random.normal(self.means,self.sigmas)




