# file matching_advertising.py

from Publisher import *
from Advertiser import *
from AdAuctionEnvironment import *
from User import *
from CTSLearner import *
from hungarian_algorithm import hungarian_algorithm, convert_matrix
import numpy as np
import matplotlib.pyplot as plt


def calculate_opt(real_q, n_slots, n_ads):
    opt = hungarian_algorithm(convert_matrix(real_q))
    m = opt[1]
    opt_q = np.array([])
    for j in range(n_slots):
        for i in range(n_ads):
            if m[i][j] == 1:
                opt_q = np.append(opt_q, real_q[i][j])
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


def samples_from_learner(cts_learner, n_ads, n_slots):
    samples = np.zeros(shape=(n_ads, n_slots))
    for i in range(N_ADS):
        for j in range(N_SLOTS):
            a = cts_learner.beta_parameters[i][j][0]
            b = cts_learner.beta_parameters[i][j][1]
            samples[i][j] = np.random.beta(a=a, b=b)
    return samples


################################################

# T - Time horizon - number of days
T = 365

number_of_experiments = 10

# number of advertisers for each publisher

N_ADS = 4
N_SLOTS = 4
N_USERS = 10  # number of users for each day
N_KLASSES = 3

publisher1 = Publisher(n_slots=4)

publishers = [publisher1]

k_p = generate_klasses_proportion(N_KLASSES)
assert k_p.sum() == 1.0
print("User klasses proportion:")
print(k_p)

real_q_klass = []
for klass in range(N_KLASSES):
    real_q_klass.append(np.random.uniform(size=(N_ADS, N_SLOTS)))

real_q_aggregate = np.sum(list(map(lambda x: x[1] * k_p[x[0]], enumerate(real_q_klass))), axis=0)

cts_rewards_per_experiment_aggregate = []
cts_rewards_per_ex_klass = [[] for i in range(N_KLASSES)]

for publisher in publishers:
    advertisers = []
    for i in range(N_ADS):
        advertiser = Advertiser(bid=1, publisher=publisher)
        advertisers.append(advertiser)

    for e in range(number_of_experiments):
        print(np.round((e + 1) / number_of_experiments * 10000) / 100, "%")
        cts_learner_aggregate = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)

        learners_by_klass = []
        for klass in range(N_KLASSES):
            learner_by_klass = CTSLearner(n_ads=N_ADS, n_slots=publisher.n_slots, t=T)
            learners_by_klass.append(learner_by_klass)

        for t in range(T):
            users = generate_users(k_p, N_USERS)
            environment = AdAuctionEnvironment(advertisers, publisher, users, real_q=real_q_aggregate,
                                               real_q_klass=real_q_klass)

            for user in users:
                # ############ aggregate Learner 
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                samples_aggregate = samples_from_learner(cts_learner_aggregate, N_ADS, N_SLOTS)
                superarm_aggregate = publisher.allocate_ads(samples_aggregate)
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward_aggregate = environment.simulate_user_behaviour_as_aggregate(user, superarm_aggregate)
                # 3. UPDATE BETA DISTRIBUTIONS
                cts_learner_aggregate.update(superarm_aggregate, reward_aggregate, t=t)

                # ######## learner for klass
                # 1. FOR EVERY ARM MAKE A SAMPLE  q_ij - i.e. PULL EACH ARM
                klass_learner = learners_by_klass[user.klass]
                klass_samples = samples_from_learner(klass_learner, N_ADS, N_SLOTS)
                superarm = publisher.allocate_ads(klass_samples)
                # 2. PLAY SUPERARM -  i.e. make a ROUND
                reward = environment.simulate_user_behaviour(user, superarm)
                # 3. UPDATE BETA DISTRIBUTIONS
                klass_learner.update(superarm, reward, t=t)

                # Set zero reward for learners except learner for user.klass
                l_ex_u_k = [x[1] for x in enumerate(learners_by_klass) if x[0] != user.klass]
                for learner in l_ex_u_k:
                    learner.collected_rewards[t].append(np.array([0, 0, 0, 0]))

        # collect results for publisher
        avg_rew_per_days_aggregate = list(map(lambda rews_day: np.mean(rews_day or [np.array([0, 0, 0, 0])], axis=0),
                                              cts_learner_aggregate.collected_rewards))
        cts_rewards_per_experiment_aggregate.append(avg_rew_per_days_aggregate)

        for klass in range(N_KLASSES):
            collected_rewards = learners_by_klass[klass].collected_rewards
            avg_rew_per_days = list(
                map(lambda rews_day: np.mean(rews_day or [np.array([0, 0, 0, 0])], axis=0), collected_rewards))
            cts_rewards_per_ex_klass[klass].append(avg_rew_per_days)

    # Plot curve

    cts_rewards_per_experiment_aggregate = np.array(cts_rewards_per_experiment_aggregate)
    for klass in range(N_KLASSES):
        rewards = cts_rewards_per_ex_klass[klass]
        cts_rewards_per_ex_klass[klass] = np.array(rewards)

    opt_q_aggregate = calculate_opt(real_q_aggregate, n_slots=N_SLOTS, n_ads=N_ADS)
    # print(opt_q_aggregate)

    cumsum_aggregate = np.cumsum(np.mean(opt_q_aggregate - cts_rewards_per_experiment_aggregate, axis=0), axis=0)

    cumsum_klass = []
    opt_q_klass = []
    for klass in range(N_KLASSES):
        opt_q_klass.append(calculate_opt(real_q_klass[klass], n_slots=N_SLOTS, n_ads=N_ADS))
        cumsum_klass.append(np.cumsum(np.mean(opt_q_klass[klass] - cts_rewards_per_ex_klass[klass], axis=0), axis=0))

    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("Reward")
    colors = ['r', 'g', 'b']
    plt.plot(list(map(lambda x: np.sum(x), cumsum_aggregate)), 'm')
    klass_rewards_per_day = []
    for klass in range(N_KLASSES):
        klass_reward_per_day = list(map(lambda x: np.sum(x), cumsum_klass[klass]))
        klass_rewards_per_day.append(klass_reward_per_day)
        plt.plot(klass_reward_per_day, colors[klass])

    plt.plot(np.sum(klass_rewards_per_day, axis=0), 'orange')

    plt.legend(["Aggregated", "Klass 1", "Klass 2", "Klass 3", "Disaggregated"])
    plt.show()
