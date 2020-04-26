# file matching_advertising.py

from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *
from User import *
from CTSLearner import *
import numpy as np
import matplotlib.pyplot as plt

# T - Time horizon
T = 365

number_of_experiments = 10

# number of advertisers for each publisher
N_ADS = 4
publisher1 = Publisher(n_slots=4)

publishers = [publisher1]

cts_rewards_per_experiment = []

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=np.random.randint(1), publisher=publisher)
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        cts_learner = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots)
        for t in range(T):
            print(t)
            print("\n")
            users = []
            N_USERS = 100  # TODO Get N_USERS from some distribution
            for i in range(N_USERS):
                user = User(feature1=np.random.binomial(1, 0.5),
                            feature2=np.random.binomial(1, 0.5),
                            klass=np.random.randint(3))
                users.append(user)

            environment = AdAuctionEnvironment(advertisers, publisher, users)
            print("CTS Step 1\n")
            # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
            for A in range(N_ADS):
                advertisers[A].q = np.random.beta(a=cts_learner.beta_parameters[A, :, 0],
                                                  b=cts_learner.beta_parameters[A, :, 1],
                                                  size=publisher.n_slots)

            # Then we choose the superarm with maximum sum reward (obtained from publisher)
            superarm = publisher.allocate_ads(advertisers)

            print("CTS Step 2\n")
            # 2. PLAY SUPERARM -  i.e. make a ROUND
            reward = environment.simulate_users_behaviour(superarm)

            print("CTS Step 3\n")
            # 3. UPDATE BETA DISTRIBUTIONS
            cts_learner.update(superarm, reward)

        # collect results for publisher
        cts_rewards_per_experiment.append(cts_learner.collected_rewards)

    # Plot graphics for
    opt = 0
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(cts_rewards_per_experiment, axis=0)), 'r')
    plt.legend(["TS"])
    plt.show()
