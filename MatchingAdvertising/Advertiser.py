import numpy as np


class Advertiser:
    def __init__(self, bid, publisher):
        self.bid = bid
        self.q = np.zeros(shape=publisher.n_slots)
        self.publisher = publisher
