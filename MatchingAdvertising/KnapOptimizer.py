import numpy as np


class KnapOptimizer():
    def __init__(self, n_bids, n_budget, n_subcampaign, bids):
        self.bids = bids
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
        # superarm
        # [row,col] subcampaign 1
        # [row,col] subcampaign 2
        # [row,col] subcampaign 3
        # [row,col] subcampaign 4

    def Optimize(self, n):
        #print(self.step1n)
        self.step1n = n
        #print(self.step1n)
        #print("Initial matrix", self.step1n, "\n")
        maxc = np.zeros(shape=(4, 4))
        for i in range(self.n_subcampaign):
            maxc[i] = np.amax(self.step1n[i], axis=1)
        self.step2n = maxc #array with the maximum value of each row of the initial matrix
        #print(self.step2n, "STEP2n")
        superarm = self.step1(self.step2n)

        return superarm

    def step1(self, step2bs):

        self.finalm[0] = step2bs[0]  #prima era self.finalm[0] = step2bs[0]
        for i in range(1, self.n_budget):
            for j in range(1, self.n_budget + 1):
                inv = self.finalm[i - 1][0:j]
                inv = inv[::-1]
                res = np.add(inv, step2bs[i][0:j])
                self.finalm[i][j - 1] = np.amax(res)
        #print("Final matrix",self.finalm, "\n")
        self.result = np.argmax(self.finalm, axis=1) #indici NELLA MATRICE DEI MASSSIMi dei valori massimi di quella riga
        # print("INIT MATRIX \n ", self.step1n)
        # print("\n-------------\n")
        # print("Step2n MATRIX ,matrice dei massimi\n ", self.step2n)
        # print("\n-------------\n")
        # print("FINAL MATRIX \n", self.finalm)
        # print("\n-------------\n")
        # print("Vettore con gli indici dei massimi di ogni riga della colonna finale\n ", self.result)
        superarm = np.zeros(shape = (4,2), dtype = int)
        bids  = []

        for i in range(self.n_subcampaign):
            val = self.step2n[i][self.result[i]] #lo troviamo nella riga i-esima della matrice 2 e nella colonna data dall'indice dei massimi della matrice secondaria
            #print("Valore", val) # valore che ottimizza la subcampain
            index = np.where(self.step1n[i][self.result[i]] == val)
            bids.append(float(self.bids[index[0]]))
            #print("INDICI", index[0]) #ora devo prendere il bid associato a quest'indice
            #print ("Bid", self.bids[index[0]])

            superarm[i] = self.result[i], index[0]
            #print("MAXVC", self.step2n)
        #print("INDICE", superarm)
        return superarm
        #print("BIDS OTTIMALI\n",bids)
