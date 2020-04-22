import numpy as np


class Publisher:
    def __init__(self, n_ad_slots):
        if not n_ad_slots > 3:
            raise SystemExit("Number of slots should be greater than 3")
        self.n_ad_slots = n_ad_slots
        self.slots = np.array([[] for i in range(n_ad_slots)])  # or just [] or np.array[]

    def allocate_ads(self):
        # here we allocate the ads to our slots
        return
