# Course Part 3
import numpy as np

# function fun maps the bid to corresponding number of clicks
def fun(x):
    return 100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))


class BiddingEnvironment():
    # at each day we have to select bid (we have 20 possible bids) in order to maximize the number of clicks
    # since we have a daily fixed budget the optimal bid is not necessary the one with the argue? value
    #   value of bids corresponds to number of clicks
    #   we can explore this correlation to speed up the learning

    # the environment returns a stochastic reward (i.e a number of clicks) depending on the pulled arm (choosen bid)

    def __init__(self, bids, sigma):
        self.bids = bids
        self.means = fun(bids)
        self.sigmas = np.ones(len(bids)) * sigma

    # returns the reward (drawn from normal distribution)
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
