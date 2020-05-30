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
        index_of_winners, value_to_pay = self.auction(advertisers,n_slots)
        print(q, "not sorted")
        q = -np.sort(-q)
        print(q, "sorted")   #the first q of each row will have the higher probability of being clicked
        #only the first winner will have the possibility to chose the first slot
        #the secondo winner will choose the second and go on
        for a in index_of_winners:  #the
            for s in range(n_slots):
                allocated = q[a][index_of_winners]
        return allocated

    def auction(self,advertisers,N_SLOTS): #how the auction is hanled according to vcg
        index_of_advertiser = []
        bids = self.set_bid(advertisers)
        for i in range(len(advertisers)):
            index_of_advertiser.append(i)
        index_of_winners = [index_of_advertiser for _, index_of_advertiser in sorted(zip(bids,index_of_advertiser), key=lambda pair: pair[0])]
        #ordering the array of the advertisers accorngly to their bid
        index_of_winners = index_of_winners[:N_SLOTS] #take the first N winners
        bids[::-1].sort() #sort the array of the bids in decscengin order
        print(bids)
        value_to_pay = bids[N_SLOTS-1] #accordinly to vcg they will pay the "second highest price", in this case since we have n bid it's not the secondo but the n+1
        #the code up is suppose to have nslots but it gives index out of range since we are not using enough advertisers
        return index_of_winners,value_to_pay

    def set_bid(self,advertisers):
        bids = []
        for a in advertisers:
            bids.append(a.budget/np.random.uniform(20,50)) #each advertiser makes a small bid compared to his budget
        return bids



   # def return_superam(self):
        #[[3, 0], [2, 1], [1, 2], [0, 3]]
