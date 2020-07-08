from KnapOptimizer import *

from Publisher import *
from Advertiser import *
from VCG_auction import *
from AdAuctionEnvironment import *
from AdvLearner import *
from User import *
from CTSLearner import *
from BiddingEnvironment import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt


def samples_from_learner(cts_learner, n_ads, n_slots):
    samples = np.zeros(shape=(n_ads, n_slots))
    for i in range(N_ADS):
        for j in range(N_SLOTS):
            a = cts_learner.beta_parameters[i][j][0]
            b = cts_learner.beta_parameters[i][j][1]
            samples[i][j] = np.random.beta(a=a, b=b)
    return samples

def calculate_opt(real_q, n_slots, n_ads):
    opt = hungarian_algorithm(convert_matrix(real_q))
    #print("CALCULATE OPT", opt)
    m = opt[1]
    opt_q = np.array([])
    for j in range(n_slots):
        for i in range(n_ads):
            if m[i][j] == 1:
                opt_q = np.append(opt_q, real_q[i][j])
    return opt_q

def calculate_opt_advreal(real_q, n_slots, n_ads):
    opt = hungarian_algorithm(convert_matrix(real_q))
    #print("CALCULATE OPT", opt)
    m = opt[1]
    opt_q = np.array([])
    for j in range(n_slots):
        if m[0][j] == 1:
            opt_q = np.append(opt_q, real_q[0][j])
    return opt_q

def generate_klasses_proportion(n_klasses):
    p = np.random.randint(100, size=n_klasses) + 20
    p = p / p.sum()
    return p


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

T = 100
number_of_experiments = 1

# number of advertisers for each publisher
N_BIDS = 4
N_BUDGET = 4
N_SUBCAMPAIGN = 4
N_ARMS = N_BIDS * N_BUDGET
N_ADS = 4
N_SLOTS = 4
N_USERS = 20  # number of users for each day
N_KLASSES = 3
N_AUCTION = 20
bids = np.linspace(start = 25, stop = 100, num = N_BIDS)
paying = 0
no_money = False

SLOTS_QUALITY = -np.sort(-np.random.choice(range(20), 4, replace=False))
publisher1 = Publisher(n_slots=4)

k_p = generate_klasses_proportion(N_KLASSES)
assert k_p.sum() == 1.0
print("User klasses proportion:")
print(k_p)

real_q_klass = []
for klass in range(N_KLASSES):
    real_q_klass.append(np.random.uniform(size=(N_ADS, N_SLOTS)))

real_q_aggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(real_q_klass))), axis=0)
opt_q_klass = list(map(lambda x: calculate_opt(x, n_slots=N_SLOTS, n_ads=N_ADS), real_q_klass))

real_q_aggregate = real_q_klass[0]
cts_rewards_per_experiment_aggregate = []
#cts_rewards_per_ex_klass = [[] for i in range(N_KLASSES)]
opt_q_adv = calculate_opt_advreal(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
opt_q_disaggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(opt_q_klass))), axis=0)


publishers = [publisher1]
k_p = generate_klasses_proportion(N_KLASSES)
vincitori = [0, 0, 0 ,0]

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=bids[np.random.randint(0,3)], publisher=publisher, budget=np.random.uniform(5000, 7000), d_budget=np.random.uniform(500, 1000))
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        total = 0
        for a in range(len(advertisers)):
            advertisers[a].budget = np.random.uniform(5000, 7000) #at every experiment i set a new budget

        print(np.round((e + 1) / number_of_experiments * 10000) / 100, "%")
        cts_learner_aggregate = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
        learner_by_subcampaign = []
        for subcampaign in range(N_SUBCAMPAIGN):
            advlearner = AdvLearner(n_arms =N_ARMS,n_ads = N_ADS,n_bids = N_BIDS, n_budget= N_BUDGET, t = T)
            learner_by_subcampaign.append(advlearner)

        knap = KnapOptimizer(n_bids=N_BIDS, n_budget=N_BUDGET, n_subcampaign=4,bids = bids)

        for t in range(T):
            print("Day:",t)
            for a in range(len(advertisers)):
                advertisers[a].d_budget = np.random.uniform(500, 1000)  # at every day i set a new d_budget
            if(t % 20 == 0):
                print("day n: ", t)
            users = generate_users(k_p,N_USERS)
            Adenvironment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate, real_q_klass=real_q_klass)

            sample_n = []
            res_auction = []
            reward_gaussian = [0, 0, 0, 0]
            q_adv0 = [0, 0, 0, 0]
            for i in range(N_SUBCAMPAIGN):
                sample_n.append(np.reshape(learner_by_subcampaign[i].estimate_n(), (4,4)))

            superarm = knap.Optimize(sample_n) #combination  optimal bid/budget for each subcampaign

            for arm in range(N_SUBCAMPAIGN):
                auction = VCG_auction(real_q_aggregate, superarm[arm], N_SLOTS, advertisers)
                res_auction.append(auction.choosing_the_slot(real_q_aggregate, SLOTS_QUALITY))
                #print(res_auction)

                idx = res_auction[arm][0].index(0)
                if(arm == 0):
                    vincitori[idx] += 1
                q_adv0[arm] = res_auction[arm][1][idx]

                paying = res_auction[arm][2]  #how much to pay for pay per click !!!!!!!!!!!
               # print(paying, "PAY")

            for user in users:
                for j in range(N_SUBCAMPAIGN):
                    reward = Adenvironment.simulate_user_behaviour_auction(user, q_adv0[j], paying, advertisers) #SIMULART |||||||||||||
                    reward_gaussian[j] += reward
                    learner_by_subcampaign[j].update_reward(reward,t)

            for i in range(N_SUBCAMPAIGN):
                 learner_by_subcampaign[i].update(superarm[i], reward_gaussian[i], t)

        print("REWARD", learner_by_subcampaign[0].collected_rewards)
        cts_rewards_per_experiment_aggregate.append(learner_by_subcampaign[0].collected_rewards)
        print(vincitori)

    cts_rewards_per_experiment_aggregate = np.array(cts_rewards_per_experiment_aggregate)
    opt_q_aggregate = 1

    cumsum_aggregate = np.cumsum(np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0),axis=0)
    cumsum_aggregate2 = (np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0))

    array_agg = (list(map(lambda x: np.sum(x), cumsum_aggregate2)))

    array_tot = []
    array_sum = []


    array_sum = np.cumsum(array_tot)

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
