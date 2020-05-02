from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users

    def simulate_users_behaviour(self, edges):
        amount_of_clicks = np.zeros(len(edges))
        # Reward for Ad with max number of clicks will be 1, and reward with 0 num of clicks will be 0
        max_clicks = 0
        for u_num in range(len(self.users)):
            # TODO users select ad according to their features or some distribution
            for edge in edges:
                i = edge[0]  # number of advertiser
                j = edge[1]  # number of slot
                # if weight of edge (i:j) is b_i * q_ij, then q_ij = w_ij/ b_i
                real_q_ij = self.advertisers[i].q[j] / self.advertisers[i].bid
                if np.random.binomial(1, real_q_ij) == 1:
                    amount_of_clicks[j] += 1
                    if amount_of_clicks[j] > max_clicks:
                        max_clicks = amount_of_clicks[j]
        print(amount_of_clicks.sum())
        print(amount_of_clicks)
        if max_clicks != 0:
            amount_of_clicks /= max_clicks
        print(amount_of_clicks.sum())
        print(amount_of_clicks)
        return amount_of_clicks
