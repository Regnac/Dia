from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users, real_q):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users
        self.real_q = real_q

    def simulate_user_behaviour(self, user, edges):
        reward = np.zeros(len(edges))
        for edge in edges:
            i = edge[0]  # number of advertiser
            j = edge[1]  # number of slot
            q_ij = self.real_q[i][j]  # real probability of click
            reward[j] = np.random.binomial(1, q_ij)
        return reward
