from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users

    def round(self, pulled_arm):
        # publisher.allocate_ads
        # simulate clicks (users behaviour)
        # return reward
        return
