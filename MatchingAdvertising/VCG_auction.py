

class VCG_auction():

    def auction(self, bids,advertisers,N_SLOTS):
        winners = list[[advertisers for _,advertisers in sorted(zip(bids,advertisers))]: N_SLOTS] #take the higher paying n advertiser
        value_to_pay = bids[N_SLOTS+1] #accordinly to vcg they will pay the "second highest price", in this case since we have n bid it's not the secondo but the n+1
        return winners,value_to_pay

