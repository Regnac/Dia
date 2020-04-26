from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users

    def simulate_users_behaviour(self, edges):
        amount_of_clicks = np.zeros(len(edges))
        for i in range(len(self.users)):
            # TODO users select ad according to their features or some distribution
            clicked_ad_number = np.random.randint(len(edges))
            amount_of_clicks[clicked_ad_number] += 1
        amount_of_clicks /= len(self.users)
        print(amount_of_clicks)
        return amount_of_clicks
