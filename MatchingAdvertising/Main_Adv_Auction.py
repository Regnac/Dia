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
from tqdm import tqdm

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

def calculate_opt_advreal(real_q):
    # opt = hungarian_algorithm(convert_matrix(real_q))
    # #print("CALCULATE OPT", opt)
    # m = opt[1]
    # opt_q = np.array([])
    # for j in range(n_slots):
    #     if m[0][j] == 1:
    #         opt_q = np.append(opt_q, real_q[0][j])
    opt_q = np.max(real_q[0])
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

def update_budget(reward, advertisers, idx_subcampaign):
    for a in range(N_ADS):
        if(reward[a] == 1):
            #print("QUANTO per sub", idx_subcampaign , "ed adv ", a, "prezzo ", paying[idx_subcampaign][a])
            #print(advertisers[a].budget , advertisers[a].d_budget )
            if(advertisers[a].budget > paying[idx_subcampaign][a]):
                advertisers[a].budget -= paying[idx_subcampaign][a]  # TOTAL BUDGET is updated
            else:
                reward[a] == 0
                #advertisers[a].budget = 0

            if(advertisers[a].d_budget[idx_subcampaign] >paying[idx_subcampaign][a]):
                advertisers[a].d_budget[idx_subcampaign] -= paying[idx_subcampaign][a]  # daily b
            else:
                reward[a] == 0
                #advertisers[a].d_budget[idx_subcampaign] = 0

            if(a == 0):
                b1.append(advertisers[a].budget)
            if (a == 1):
                b2.append(advertisers[a].budget)
            if (a == 2):
                b3.append(advertisers[a].budget)
            if (a == 3):
                b4.append(advertisers[a].budget)
            #if(a == 0):
                #print(advertisers[a].d_budget)
        #if (a == 0):  # Test
            #print(advertisers[a].budget, "Total budget of", a)

            #print("todo")  # TODO   set real_q = 0 so people wont click on that ad !!!!!!!!!!!!!!!!!!!!!!!!

def check_dbudget(advertisers,idx_sub):
    for a in range(N_ADS):
        if (advertisers[a].d_budget[idx_sub] <= 0):
            no_money_d[idx_sub][a] = True

def check_budget(advertisers):
    for a in range(N_ADS):
        if (advertisers[a].budget <= 0):
            no_money_b[a] = True
            for i in range(N_SUBCAMPAIGN):
                no_money_d[i][a] = True

def get_q(res_auction,arm):
    for i in range(N_ADS):
        idx = res_auction[arm][0].index(i)
        q_adv[arm][i] = res_auction[arm][1][idx]
    #print("res auction ",res_auction[arm][0])
    idx = res_auction[arm][0].index(0)
    #print("position auction")
    vincitori[idx] += 1
    return q_adv[arm]
    #q_adv0[arm] = res_auction[arm][1][idx]

T = 200
number_of_experiments = 5

# number of advertisers for each publisher
N_BIDS = 4
N_BUDGET = 4
N_SUBCAMPAIGN = 4
N_ARMS = N_BIDS * N_BUDGET
N_ADS = 4
N_SLOTS = 4
N_USERS = 20  # number of users for each day
N_KLASSES = 3
N_AUCTION = 5
tot_b = 2500000
bids = np.linspace(start = 25, stop = 100, num = N_BIDS)
dbud = (tot_b/T) / 4
d_budget= [dbud *3, dbud *3, dbud *3, dbud *3]
#our_d_budget = [2500, 5000, 7500, 10000]
our_d_budget = [dbud, dbud * 2, dbud * 3, dbud * 4]
paying = np.zeros(shape=(4,4))
no_money_d = np.zeros((4, 4), dtype=bool)
no_money_b = np.zeros((4), dtype=bool)

print("D_budget", d_budget, "OUR D_BUDGET", our_d_budget)
b1 = []
b2 = []
b3 = []
b4 = []

SLOTS_QUALITY = -np.sort(-np.random.choice(range(5), 4, replace=False))
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
cts_rewards_per_experiment_aggregate = [[], [], [], []]
#cts_rewards_per_ex_klass = [[] for i in range(N_KLASSES)]
#opt_q_adv = calculate_opt_advreal(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
opt_q_disaggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(opt_q_klass))), axis=0)


publishers = [publisher1]
k_p = generate_klasses_proportion(N_KLASSES)
vincitori = [0, 0, 0 ,0]

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=bids[np.random.randint(0,3)], publisher=publisher, budget=tot_b, d_budget= np.random.uniform(d_budget[np.random.randint(0,3)], size=4 ))
        advertisers.append(advertiser)


    for e in range(number_of_experiments):
        total = 0
        for a in range(len(advertisers)):
            advertisers[a].budget = tot_b #at every experiment i set a new budget

        print(np.round((e + 1) / number_of_experiments * 10000) / 100, "%")
        cts_learner_aggregate = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
        learner_by_subcampaign = []
        for subcampaign in range(N_SUBCAMPAIGN):
            advlearner = AdvLearner(n_arms =N_ARMS,n_ads = N_ADS,n_bids = N_BIDS, n_budget= N_BUDGET, t = T, bids = bids, D_budget = our_d_budget)
            learner_by_subcampaign.append(advlearner)

        knap = KnapOptimizer(n_bids=N_BIDS, n_budget=N_BUDGET, n_subcampaign=4,bids = bids)
        for t in tqdm(range(T)):
            no_money_d = np.zeros((4, 4), dtype=bool)
            #print("Day:",t)
            for a in range(1, len(advertisers)):  #DAYLI!!!!!!!!!!!!!!!
                advertisers[a].d_budget = np.random.uniform(d_budget[np.random.randint(0,3)], size=4)  # at every day i set a new d_budget
            #if(t % 20 == 0):
                #print("day n: ", t)
            users = generate_users(k_p,N_USERS)
            Adenvironment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate, real_q_klass=real_q_klass)

            ####### INITIALIZE VARIABLE
            sample_n = []
            res_auction = []
            reward_gaussian = [0, 0, 0, 0]
            q_adv = np.zeros(shape=(4,4))
            ####### SAMPLE + OPTIMIZIATION
            for i in range(N_SUBCAMPAIGN):
                sample_n.append(np.reshape(learner_by_subcampaign[i].estimate_n(), (4,4)))

            superarm = knap.Optimize(sample_n) #combination  optimal budget/bid for each subcampaign
            #print(superarm)
            for i in range(N_SUBCAMPAIGN):
                advertisers[0].d_budget[i] = our_d_budget[superarm[i][0]]

            #print(advertisers[0].d_budget)
            # for i in range(N_SUBCAMPAIGN):
            #     advertisers[0].d_budget[i] = superarm[i]

            ####### AUCTION
            for auc in range(N_AUCTION):
                for arm in range(N_SUBCAMPAIGN):
                    auction = VCG_auction(real_q_aggregate, superarm[arm], N_SLOTS, advertisers)
                    res_auction.append(auction.choosing_the_slot(real_q_aggregate, SLOTS_QUALITY,arm))
                    q_adv[arm] = get_q(res_auction,arm)
                    check_dbudget(advertisers, arm)

                    paying[arm] = [x for _,x in sorted(zip(res_auction[arm][0],res_auction[arm][2]))]
                    #print(paying)#how much to pay for pay per click !!!!!!!!!!!
                   # print(paying, "PAY")
                check_budget(advertisers)

                for user in users:

                    for j in range(N_SUBCAMPAIGN):
                        for adv in range(N_ADS):
                            if(no_money_d[j][adv]):
                                q_adv[j][adv] = 0
                                #print("Day: ", t, " Adv: ", adv, "Subcampaign: ", j)
                        reward = Adenvironment.simulate_user_behaviour_auction(user, q_adv[j], advertisers) #SIMULART |||||||||||||
                        update_budget(reward, advertisers,j)
                        check_dbudget(advertisers,j)
                        check_budget(advertisers)
                        reward_gaussian[j] += reward[0]
                        learner_by_subcampaign[j].update_reward(reward[0],t)

            for i in range(N_SUBCAMPAIGN):
                learner_by_subcampaign[i].update(superarm[i], reward_gaussian[i], t)
                #print("adv0 d", advertisers[0].d_budget)
                #print("adv b", advertisers[0].budget)
        print("REWARD", learner_by_subcampaign[0].collected_rewards)
        for i in range(N_SUBCAMPAIGN):
            cts_rewards_per_experiment_aggregate[i].append(learner_by_subcampaign[i].collected_rewardsy)
        print(vincitori)

    print("ADV 0 ", advertisers[0].budget,"ADV 1 ", advertisers[1].budget," ADV 2 ", advertisers[2].budget,"ADV 3 ", advertisers[3].budget)
    for i in range(N_SUBCAMPAIGN):
        cts_rewards_per_experiment_aggregate[i] = np.array(cts_rewards_per_experiment_aggregate[i])
    opt_q_aggregate = calculate_opt_advreal(real_q_aggregate)
    cumsum_aggregate = []
    for i in range(N_SUBCAMPAIGN):
        cumsum_aggregate.append(np.cumsum(np.mean(opt_q_aggregate - (cts_rewards_per_experiment_aggregate[i]/(N_USERS*N_AUCTION)), axis=0),axis=0))
    #cumsum_aggregate2 = (np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0))

    #array_agg = (list(map(lambda x: np.sum(x), cumsum_aggregate2)))

    array_tot = []
    array_sum = []
    array_sum = np.cumsum(array_tot)

    #plt.plot(list(map(lambda x: np.sum(x), cumsum_aggregate[0])), 'm')
    #  plt.plot(list(map(lambda x: np.sum(x), cumsum_disaggregate)), 'orange')
    plt.figure(4)
    plt.title("Budget")
    plt.xlabel("t")
    plt.ylabel("Budget")
    plt.plot(b1, 'm')
    plt.plot(b2, 'r')
    plt.plot(b3, 'b')
    plt.plot(b4, 'y')
    #plt.legend(["Aggregated"])
    #plt.show()
    # plt.plot(b4, 'y')
    plt.figure(1)
    plt.title("Regret")
    plt.plot(cumsum_aggregate[0], "m")
    plt.plot(cumsum_aggregate[1], "g")
    plt.plot(cumsum_aggregate[2], "b")
    plt.plot(cumsum_aggregate[3], "y")

    plt.legend(["Sub1", "Sub2", "Sub3", "Sub4"])
    plt.figure(2)
    plt.title("Reward")
    plt.plot(learner_by_subcampaign[0].collected_rewardsy / N_USERS, "m")
    plt.plot(learner_by_subcampaign[1].collected_rewardsy / N_USERS, "g")
    plt.plot(learner_by_subcampaign[2].collected_rewardsy / N_USERS, "b")
    plt.plot(learner_by_subcampaign[3].collected_rewardsy / N_USERS, "y")
    #plt.legend(["Sub1", "Sub2", "Sub3", "Sub4"])
    #plt.plot(learner_by_subcampaign[0].collected_rewardsy, "m")
    #plt.plot(learner_by_subcampaign[1].collected_rewardsy , "b")
    #plt.plot(learner_by_subcampaign[2].collected_rewardsy, "g")
    #plt.plot(learner_by_subcampaign[3].collected_rewardsy, "y")
    plt.legend(["Sub1", "Sub2","Sub3","Sub4"])
    plt.figure(3)
    plt.title("Original REGRET")
    plt.xlabel("t")
    plt.ylabel("Regret")
    # colors = ['r', 'g', 'b']
    # colors = ['r']
    plt.plot(np.mean(cumsum_aggregate,axis=0), 'm')
    #plt.plot(cumsum_aggregate[1], 'g')
    #plt.plot(cumsum_aggregate[2], 'b')
    #plt.plot(cumsum_aggregate[3], 'y')

    plt.legend(["Sub1"])
    #plt.plot(np.cumsum(np.mean(N_USERS - learner_by_subcampaign[0].collected_rewardsy, axis=0)), "g")
    # plt.legend(["Aggregated", "Disaggregated","Contexed"])
    #plt.legend(["Aggregated"])
    plt.show()
