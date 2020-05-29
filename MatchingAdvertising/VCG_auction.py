import numpy as np

class VCG_auction():

    def set_bid(self,advertisers):
        bids = []
        for a in range(advertisers):
            bids.append(advertisers.budget/np.random.uniform(20,50)) #each advertiser makes a small bid compared to his budget
        return bids

    def auction(self,bids,advertisers,N_SLOTS): #how the auction is hanled according to vcg
        index_of_advertiser = []
        for i in range(advertisers):
            index_of_advertiser.append(i)
        index_of_winners = list[[index_of_advertiser for _,index_of_advertiser in sorted(zip(-bids,index_of_advertiser))]: N_SLOTS] #take the higher paying n advertiser
        bids = -np.sort(-bids) #descending order
        value_to_pay = bids[N_SLOTS+1] #accordinly to vcg they will pay the "second highest price", in this case since we have n bid it's not the secondo but the n+1
        return index_of_winners,value_to_pay

    def choosing_the_slot(self,index_of_winners,q, N_SLOTS):
        ## Advertiser 1 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 2 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 3 - [Slot1, Slot2, Slot3, Slot4]
        ## Advertiser 4 - [Slot1, Slot2, Slot3, Slot4]
        print(q, "not sorted")
        q = -np.sort(-q)
        print(q, "sorted")   #the first q of each row will have the higher probability of being clicked
        #only the first winner will have the possibility to chose the first slot
        #the secondo winner will choose the second and go on
        for a in range(index_of_winners):  #the
            for s in range(N_SLOTS):
                allocated = q[a][index_of_winners]

        return allocated
