from Learner import *


# Combinatorial Thompson Sampling Learner
class CTSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # initialize parameters of learner

    def pull_arm(self):
        # pull arm
        return

    def update(self, pulled_arm, reward):
        # update beta parameters
        # self.update_observations(pulled_arm, reward)
        return
