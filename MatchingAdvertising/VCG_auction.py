import numpy as np


class VCG_auction():
    def __init__(self, q, N_SLOTS, advertisers):
        self.q = q
        self.N_SLOTS = N_SLOTS
        self.advertisers = advertisers

    def choosing_the_slot(self, q, n_slots, advertisers):
        ## Advertiser 1 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 2 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 3 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 4 - [Slot1, Slot2, Slot3, Slot4]
        index_of_winners = self.auction(advertisers, n_slots)
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
        return allocated

    def auction(self, advertisers, N_SLOTS):  # how the auction is hanled according to vcg
        index_of_advertiser = []
        bids = [[70,56,21,7],[50,40,15,5],[10,8,3,1],[80,64,24,8]]

        for i in range(len(advertisers)):
            index_of_advertiser.append(i)
        index_of_winners = [index_of_advertiser for _, index_of_advertiser in
                            sorted(zip(bids, index_of_advertiser), key=lambda pair: pair[0])]
        # ordering the array of the advertisers accorngly to their bid
        index_of_winners = index_of_winners[:N_SLOTS]  # take the first N winners
        bids[::-1].sort()  # sort the array of the bids in decscengin order
        # print(bids)

        paying = self.value_to_pay(bids, index_of_winners[::-1])
        # the code up is suppose to have nslots but it gives index out of range since we are not using enough advertisers
        return index_of_winners


    def value_to_pay(self, bids, index_of_winners):
        ## Advertiser 1 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 2 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 3 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        ## Advertiser 4 - [Bids for Slot1,Bids for Slot2,Bids for Slot3,Bids for Slot4]
        #etc
        pay = []
        bids2 = [[[] for j in range(4)] for i in range(4)]

        print(index_of_winners)

        for i in range(4):
            bids2[i] = (bids[index_of_winners[i]])

        print(self.value_single_ad(0,0,bids2))

        return 0

    def value_single_ad(self, i,j, bids):

        Y = []
        X = np.diagonal(bids[i+1:3,0:3])
        var_Y = np.diagonal(bids[i:3,0:3])

        print(X,"  ",var_Y)

        return 0





