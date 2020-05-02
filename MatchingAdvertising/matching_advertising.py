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
n_slots = 4

publisher1 = Publisher(n_slots)
publishers = [publisher1]

cts_rewards_per_experiment = []

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=1, publisher=publisher)
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
                advertisers[A].sampled_weights = np.random.beta(a=cts_learner.beta_parameters[A, :, 0],
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

    # Plot graphics
    # NOW THIS OPT VALUE IS A CRUTCH. But we should determine it somehow. It MAKE influence on our plot!
    # Try to play with this value and you will see the 'normal' regret plot
    opt = np.float64(3)  # TODO understand how do we obtain opt. I'm sure we have to look at constant q_ij
    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(opt - cts_rewards_per_experiment, axis=0)), 'r')
    plt.legend(["CTS"])
    plt.show()
