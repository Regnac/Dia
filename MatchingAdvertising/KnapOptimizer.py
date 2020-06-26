import numpy as np

class KnapOptimizer():
    def __init__(self,n_bids, n_budget):
        self.n_bids = n_bids
        self.n_budget = n_budget
        self.step1n = [[[] for j in range(self.n_bids)] for i in range(self.n_budget)]
        self.step1n = np.random.randint(120, size=(self.n_bids, self.n_budget))
        self.first = True

    def step1a(self):
        # if(self.first == True):
        #     self.first =False
        # else:
        #     self.step1n = n
        print(self.step1n)
        maxc = np.amax(self.step1n, axis=1)
        print(maxc)
        self.step1b(maxc)

    def step1b(self, maxc):
        #step2bs = [[[] for j in range(self.n_bids)] for i in range(self.n_budget)]
        #step2bs[0] = maxc
        mat = np.random.randint(60, 120, size=(3, 4))
        step2bs = np.vstack((maxc, mat))
        print("\n Last step 1 matrix: \n", step2bs)
        print("\n")
        self.step2(step2bs)


    def step2(self, step2bs):
        finalm = [[[] for j in range(self.n_bids)] for i in range(self.n_budget)]
        finalm[0] = step2bs[0]
        for i in range(1,self.n_budget):
            for j in range(1, self.n_budget +1 ):
                print("-------------------")
                inv = finalm[i-1][0:j]
                inv = inv[::-1]
                print("inv: ",inv)
                res = np.add(inv, step2bs[i][0:j])
                print("res:",res)
                finalm[i][j-1] = np.amax(res)
                # print("x: ", x, "y; ", y,"i:", i)
                # print("riga prima",finalm[i-1][x])
                # print("sovrainpressione",step2bs[i][y])
                #np.append(sum, finalm[i-1][x] + step2bs[i][y])
                #print(sum)
                print("END-----------------")
        print(finalm)
        print(np.argmax(finalm, axis=1))
