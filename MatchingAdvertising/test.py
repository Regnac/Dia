from KnapOptimizer import *

from Publisher import *
from Advertiser import *
from VCG_auction import *
from AdAuctionEnvironment import *
from AdvLearner import *
from User import *
from CTSLearner import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt

def generate_users(klasses_proportion, n_users):
    users = []
    klasses_features = [
        [1, 1],
        [0, 1],
        [1, 0]
    ]
    klasses = np.random.choice([0, 1, 2], n_users, p=klasses_proportion)
    for klass in klasses:
        f1 = klasses_features[klass][0]
        f2 = klasses_features[klass][1]
        user = User(feature1=f1, feature2=f2, klass=klass)
        users.append(user)
    np.random.shuffle(users)
    return users


T = 2

number_of_experiments = 40

# number of advertisers for each publisher
N_BIDS = 4
N_BUDGET = 4
N_ARMS = N_BIDS * N_BUDGET
N_ADS = 4
N_SLOTS = 4
N_USERS = 4  # number of users for each day
N_KLASSES = 3
bids = np.linspace(0,1, N_ARMS)

SLOTS_QUALITY = -np.sort(-np.random.choice(range(20), 4, replace=False))

publisher1 = Publisher(n_slots=4)

publishers = [publisher1]

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=np.random.randint(1, 10), publisher=publisher, budget=np.random.uniform(1, 100))
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        print(np.round((e + 1) / number_of_experiments * 10000) / 100, "%")
        #cts_learner_aggregate = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
        advlearner = AdvLearner(n_arms =N_ARMS, arms=bids,n_bids = N_BIDS, n_budget= N_BUDGET, t = T)
        #learners_by_klass = []
        # for klass in range(N_KLASSES):
        #     learner_by_klass = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
        #     learners_by_klass.append(learner_by_klass)
        knap = KnapOptimizer(n_bids=4, n_budget=4)

        for t in range(T):
            users = generate_users(1, N_USERS)
            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate,
                                               real_q_klass=real_q_klass)
            n = advlearner.estimate_n()
            optbid = knap.step1a(n)
            auction = VCG_auction(real_q_aggregate, N_SLOTS, advertisers)
            q_ij = auction.choosing_the_slot(real_q_aggregate, advertisers, SLOTS_QUALITY)

            for user in users:
                # ############ aggregate Learner
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                samples_aggregate = samples_from_learner(cts_learner_aggregate, N_ADS, N_SLOTS)
                # superarm_aggregate = publisher.allocate_ads(samples_aggregate, advertisers,real_q_aggregate) #######################################

                # 2. PLAY SUPERARM -  i.e. make a ROUND

                reward_aggregate = environment.simulate_user_behaviour_bidding(q_ij)
                # print(reward_aggregate, "REWARD")

                # 3. UPDATE BETA DISTRIBUTIONS
                # cts_learner_aggregate.update(superarm_aggregate, reward_aggregate, t=t)
                cts_learner_aggregate.update_after_auction(reward_aggregate, t)

        # collect results for publisher
        cts_rewards_per_experiment_aggregate.append(cts_learner_aggregate.collected_rewards)

        # for klass in range(N_KLASSES):
        #     collected_rewards = learners_by_klass[klass].collected_rewards
        #     cts_rewards_per_ex_klass[klass].append(collected_rewards)

    # Plot curve
    # Prepare data for aggregated model
    cts_rewards_per_experiment_aggregate = np.array(cts_rewards_per_experiment_aggregate)
    opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
    # print(opt_q_aggregate, "OPT")
    # print(cts_rewards_per_experiment_aggregate, "REW")
    cumsum_aggregate = np.cumsum(np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0), axis=0)

    # Join disaggregated rewards for each experiment and day
    # cts_rewards_per_experiment_disaggregate = np.zeros(shape=np.shape(cts_rewards_per_experiment_aggregate))
    # for ex in range(number_of_experiments):
    #     for t in range(T):
    #         c = []
    #         for klass in range(N_KLASSES):
    #             c.append(cts_rewards_per_ex_klass[klass][ex][t])
    #
    #         cts_rewards_per_experiment_disaggregate[ex][t] = np.sum(np.array(c), axis=0)

    #  opt_q_klass = list(map(lambda x: calculate_opt(x, n_slots=N_SLOTS, n_ads=N_ADS), real_q_klass))

    # opt_q_disaggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(opt_q_klass))), axis=0)

    #  cumsum_disaggregate = np.cumsum(np.mean(opt_q_disaggregate - cts_rewards_per_experiment_disaggregate, axis=0),axis=0)
    # cumsum_disaggregate2 = (np.mean(opt_q_disaggregate - cts_rewards_per_experiment_disaggregate, axis=0))
    cumsum_aggregate2 = (np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0))

    #  array_dis = (list(map(lambda x: np.sum(x), cumsum_disaggregate2)))
    array_agg = (list(map(lambda x: np.sum(x), cumsum_aggregate2)))

    array_tot = []
    array_sum = []

    # for t in range(T):
    #     if(t<7):
    #         array_tot.append(array_agg[t])
    #     if (t>=7):
    #         array_tot.append(array_dis[t])  #maybe here is t-7

    array_sum = np.cumsum(array_tot)

    # print(array_agg, "AGG")
    # print(array_dis, "DIS")
    # print(array_tot, "TOT")
    # print(array_sum, "SUM")

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Regret")
    # colors = ['r', 'g', 'b']
    colors = ['r']
    plt.plot(list(map(lambda x: np.sum(x), cumsum_aggregate)), 'm')
    #  plt.plot(list(map(lambda x: np.sum(x), cumsum_disaggregate)), 'orange')
    plt.plot(array_sum, 'r')
    # plt.legend(["Aggregated", "Disaggregated","Contexed"])
    plt.legend(["Aggregated"])
    plt.show()
knap = KnapOptimizer(n_bids = 4, n_budget= 4)
knap.step1a()