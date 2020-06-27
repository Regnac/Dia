import numpy as np


class VCG_auction():
    def __init__(self, q, N_SLOTS, advertisers):
        self.q = q
        self.N_SLOTS = N_SLOTS
        self.advertisers = advertisers

    def choosing_the_slot(self, q, advertisers,slots_q):
        ## Advertiser 1 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 2 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 3 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 4 - [Slot1, Slot2, Slot3, Slot4]
        index_of_winners = self.auction(advertisers,slots_q)
       # print(q, "not sorted")
        q = -np.sort(-q)
       # print(q, "sorted")  # the first q of each row will have the higher probability of being clicked
        # only the first winner will have the possibility to chose the first slot
        # the secondo winner will choose the second and go on
      #  print(index_of_winners, "index of winners")
        allocated = []
        i = 0
        for a in index_of_winners:  # the best bidder takes the best slot for him, the second best bidder the secondo best (of his row) etc
            allocated.append(q[a][i])
            i += 1
     #   print(allocated, "allocated")
        print("index of winner ", index_of_winners)
        print("allocated ", allocated)
        return allocated

    def auction(self, advertisers, slots_q):  # how the auction is hanled according to vcg
        index_of_advertiser = []

        qv= []
        for i in range(self.N_SLOTS):
            qv.append([0, 0, 0, 0])
        lambdaeff = [1, 0.8, 0.5, 0.3]
        for i in range(self.N_SLOTS):
            for j in range(len(slots_q)):
                #bids[i][j] = advertisers[i].bid * slots_q[j] * lambdaeff[i]
                qv[i][j] = advertisers[i].bid * slots_q[j]

        print(qv)

        #bids = [[70, 56, 21, 7], [50, 40, 15, 5], [10, 8, 3, 1], [80, 64, 24, 8]]
        for i in range(len(advertisers)):
            index_of_advertiser.append(i)
        index_of_winners = [index_of_advertiser for _, index_of_advertiser in
                            sorted(zip(qv, index_of_advertiser), key=lambda pair: pair[0])]
        # ordering the array of the advertisers accorngly to their bid
        index_of_winners = index_of_winners[:self.N_SLOTS]  # take the first N winners
        qv[::-1].sort()  # sort the array of the bids in decscengin order
        # print(bids)

        paying = self.value_to_pay(qv, index_of_winners[::-1],self.N_SLOTS)
        # the code up is suppose to have nslots but it gives index out of range since we are not using enough advertisers
        return index_of_winners


    def value_to_pay(self, qv, index_of_winners,N_SLOTS):
        ## Advertiser 1 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 2 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 3 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 4 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        #etc
        pay = []
        bids2= np.zeros(shape=(4, 4))
        print(index_of_winners)

        for i in range(N_SLOTS):
            bids2[i] = (qv[index_of_winners[i]])
        print(bids2)
        for i in range(N_SLOTS):
            pay.append(self.value_single_ad(i,bids2))
        print(self.value_single_ad(3,bids2))
        print(pay)
        return pay

    def value_single_ad(self, i, bids):

        X = np.diagonal(bids[i+1:self.N_SLOTS,:], offset=i)
        var_Y = np.diagonal(bids[i+1:self.N_SLOTS, i+1:self.N_SLOTS])
        var = np.sum(X) - np.sum(var_Y)

        return var




