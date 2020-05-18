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
         #   if(user.feature1 == 1):
         #       q_ij = self.real_q[i][j] + 0.001 # real probability of click
         #   if (user.feature1 != 1):
         #       q_ij = self.real_q[i][j] - 0.001  # real probability of click
         #   if (user.feature2 == 1):
         #       q_ij = self.real_q[i][j] + 0.001  # real probability of click
         #   if (user.feature2 != 1):
         #       q_ij = self.real_q[i][j] - 0.001  # real probability of click
            q_ij = self.real_q[i][j]
            reward[j] = np.random.binomial(1, q_ij)
        return reward
