# file matching_advertising.py

from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *
from User import *
from CTSLearner import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt

# T - Time horizon - number of days
T = 365

number_of_experiments = 100

# number of advertisers for each publisher

N_ADS = 4
N_SLOTS = 4
N_USERS = 10  # number of users for each day
real_q = np.random.uniform(size=(N_ADS, N_SLOTS))
print("Real q")
print(real_q)

publisher1 = Publisher(n_slots=4)

publishers = [publisher1]

cts_rewards_per_experiment = []

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=1, publisher=publisher)
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        print(np.round(e / number_of_experiments * 10000)/100, "%")
        cts_learner = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
        for t in range(T):
            users = []
            for i in range(N_USERS):
                user = User(feature1=np.random.binomial(1, 0.5),
                            feature2=np.random.binomial(1, 0.5),
                            klass=np.random.randint(3))
                users.append(user)

            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q)

            for user in users:
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                samples = np.zeros(shape=(N_ADS, N_SLOTS))
                for i in range(N_ADS):
                    for j in range(N_SLOTS):
                        samples[i][j] = np.random.beta(a=cts_learner.beta_parameters[i][j][0],
                                                       b=cts_learner.beta_parameters[i][j][1])

                # Then we choose the superarm with maximum sum reward (obtained from publisher)
                superarm = publisher.allocate_ads(samples)
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward = environment.simulate_user_behaviour(user, superarm)

                # 3. UPDATE BETA DISTRIBUTIONS
                cts_learner.update(superarm, reward, t=t)

        # collect results for publisher
        avg_rew_per_days = list(map(lambda rews_day: np.mean(rews_day or [np.array([0, 0, 0, 0])], axis=0),
                                    cts_learner.collected_rewards))
        cts_rewards_per_experiment.append(avg_rew_per_days)

    # Plot curve

    opt = hungarian_algorithm(convert_matrix(real_q))
    m = opt[1]
    opt_q = np.array([])
    for j in range(N_SLOTS):
        for i in range(N_ADS):
            if m[i][j] == 1:
                opt_q = np.append(opt_q, real_q[i][j])
    cts_rewards_per_experiment = np.array(cts_rewards_per_experiment)
    print(opt_q)

    cumsum = np.cumsum(np.mean(opt_q - cts_rewards_per_experiment, axis=0), axis=0)

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Regret")
    colors = ['c']
    plt.plot(list(map(lambda x: np.sum(x), cumsum)), 'm')
    plt.legend(["Total"])
    plt.show()
