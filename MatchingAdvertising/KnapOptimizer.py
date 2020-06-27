import numpy as np

class KnapOptimizer():
    def __init__(self,n_bids, n_budget, n_subcampaign):
        self.n_bids = n_bids
        self.n_budget = n_budget
        self.n_subcampaign = n_subcampaign
        self.step1n = []
        self.step2n = self.step1n.append([[[] for j in range(self.n_bids)] for i in range(self.n_budget)])
        for i in range(n_subcampaign):
            self.step1n.append([[[] for j in range(self.n_bids)] for i in range(self.n_budget)])
            self.step1n[i] = np.random.randint(120, size=(self.n_bids, self.n_budget))
        self.finalm = [[[] for j in range(self.n_bids)] for i in range(self.n_budget)]
        self.result = 0
        self.first = True

    def step1a(self,n):
        if(self.first == True):
            self.first = False
        else:
            self.step1n = n
        print(self.step1n)
        maxc = np.zeros(shape=(4,4))
        for i in range(self.n_subcampaign):
            maxc[i] = np.amax(self.step1n[i], axis = 1)
        self.step2n = maxc
        self.step2(self.step2n)

    def step2(self, step2bs):

        self.finalm[0] = step2bs[0]
        for i in range(1,self.n_budget):
            for j in range(1, self.n_budget +1 ):
                inv = self.finalm[i-1][0:j]
                inv = inv[::-1]
                res = np.add(inv, step2bs[i][0:j])
                self.finalm[i][j-1] = np.amax(res)
        print(self.finalm)
        self.result = np.argmax(self.finalm, axis=1)
        return self.result

    def print_all(self):
        print("SUB1 MATRIX ", self.step1n)
        print("\n-------------\n")
        print("VN MATRIX ", self.finalm)
        print("\n-------------\n")
        print("RESULT ", self.result)
        val = self.step2n[1][self.result[1]]
        index = np.where(self.step1n[1][self.result[1]] == val)
        print(index[0])
        print("MAXVC", self.step2n)
