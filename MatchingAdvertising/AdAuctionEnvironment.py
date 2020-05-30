from Environment import *


class AdAuctionEnvironment(Environment):
    #AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate, real_q_klass=real_q_klass)
    def __init__(self, advertisers, publisher, users, real_q, real_q_klass):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users
        self.real_q = real_q
        self.real_q_klass = real_q_klass

    def simulate_user_behaviour(self, user, edges):
        reward = np.zeros(len(edges))
        for edge in edges:
            i = edge[0]  # number of advertiser
            j = edge[1]  # number of slot
            q_ij = self.real_q_klass[user.klass][i][j]  # real probability of click
            reward[j] = np.random.binomial(1, q_ij)
        return reward

    def simulate_user_behaviour_as_aggregate(self, user, edges):
        reward = np.zeros(len(edges))
        for edge in edges:
            i = edge[0]  # number of advertiser
            j = edge[1]  # number of slot
            q_ij = self.real_q[i][j]  # real probability of click
            reward[j] = np.random.binomial(1, q_ij)
        return reward
