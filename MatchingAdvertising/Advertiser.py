import numpy as np


class Advertiser:
    def __init__(self, bid, publisher):
        self.bid = bid
        self.q = np.zeros(shape=publisher.n_slots)
        self.sampled_weights = np.zeros(shape=publisher.n_slots)
        for i in range(len(self.q)):
            self.q[i] = np.random.uniform()
        self.q[::-1].sort()
        print(self.q)

        self.publisher = publisher
