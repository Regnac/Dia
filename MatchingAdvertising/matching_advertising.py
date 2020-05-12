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
T = 80

number_of_experiments = 50

# number of advertisers for each publisher

N_ADS = 6
N_SLOTS = 4
N_USERS = 100  # number of users for each day
real_q = np.random.uniform(size=(N_ADS, N_SLOTS))
print("Real q")
print(real_q)


if(N_ADS != N_SLOTS):
    while (N_ADS > N_SLOTS):
        print("increase the number of coulm with dummy number")
        X0 = np.zeros((N_ADS,1)) #create a dummy column
        N_SLOTS += 1
        real_q = np.hstack((real_q, X0))  #add the dummy column

    while (N_ADS < N_SLOTS):
        print("increase the number of row with dummy number")
        X0 = np.zeros(( 1,N_SLOTS))  # create a dummy column
        N_ADS += 1
        real_q = np.vstack([real_q,X0])


publisher1 = Publisher(N_SLOTS)

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
            # print(t)
            users = []
            for i in range(N_USERS):
                user = User(feature1=np.random.binomial(1, 0.5),  #define users
                            feature2=np.random.binomial(1, 0.5),
                            klass=np.random.randint(3))
                users.append(user)  #append users

            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q) #create the envirometn

            # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
            samples = np.zeros(shape=(N_ADS, N_SLOTS))
            for i in range(N_ADS):
                for j in range(N_SLOTS):
                    #print(i,j)
                    samples[i][j] = np.random.beta(a=cts_learner.beta_parameters[i][j][0],  #update samples
                                                   b=cts_learner.beta_parameters[i][j][1])

            # Then we choose the superarm with maximum sum reward (obtained from publisher)
            superarm = publisher.allocate_ads(samples)   #THIS IS DIFFERENT THAN BEFORE

            for user in users:
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward = environment.simulate_user_behaviour(user, superarm)

                # 3. UPDATE BETA DISTRIBUTIONS
                cts_learner.update(superarm, reward)

        # collect results for publisher
        cts_rewards_per_experiment.append(cts_learner.collected_rewards)

    # Plot curve

    opt = hungarian_algorithm(convert_matrix(real_q))
    m = opt[1]
    opt_q = np.array([])
    for j in range(N_SLOTS):
        for i in range(N_ADS):
            if m[i][j] == 1:
                opt_q = np.append(opt_q, real_q[i][j])
    cts_rewards_per_experiment = np.array(cts_rewards_per_experiment)
   # print(opt_q)

    cumsum = np.cumsum(np.mean(opt_q - cts_rewards_per_experiment, axis=0), axis=0)

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Regret")
   # colors = ['r', 'g', 'b', 'c']
    #for k in range(len(cumsum[0, :])):
        #plt.plot(cumsum[:, k], colors[k])
    plt.plot(list(map(lambda x: np.sum(x), cumsum)), 'm')
  #  plt.legend(["Slot 1", "Slot 2", "Slot 3", "Slot 4", "Total"])
    plt.show()
