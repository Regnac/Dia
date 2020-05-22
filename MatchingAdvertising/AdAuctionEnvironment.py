from Environment import *
#import Context


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
            classuser = self.get_user_class(user.feature1, user.feature2)
            q_ij = self.real_q[classuser][i][j]
            reward[j] = np.random.binomial(1, q_ij)
        return reward

    def get_user_class(self,feature1,feature2):
        if (feature1 == 0 and feature2 == 0):
            return 0
        if (feature1 == 1 and feature2 == 0):
            return 1
        if (feature1 == 0 and feature2 == 1):
            return 2
        if(feature1 == 1 and feature2 == 1):
            return 3
