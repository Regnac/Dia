from KnapOptimizer import *
import numpy as np

bids = np.linspace(start = 25, stop = 100, num = 4)
knap = KnapOptimizer(n_bids=4, n_budget=4, n_subcampaign=4,bids = bids )
knap.Optimize([])
#knap.print_all()