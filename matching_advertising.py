# file matching_advertising.py

from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *

# T - Time horizon
T = 365

publishers = [publisher1, publisher2, publisher3, publisher4]

for publisher in publishers:
    environment = AdAuctionEnvironment(advertisers, publisher, users)
    cts_learner = CTSLearner()
    for t in range(T):
        cts_learner.pull_arm
        environment.round
        cts_learner.update

    # collect results for publisher

# analyse data and plot graphics
