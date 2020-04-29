import numpy as np
from hungarian_algorithm import convert_matrix as cm
from hungarian_algorithm import hungarian_algorithm as hungarian_algorithm_max


class Publisher:
    def __init__(self, n_slots):
        if n_slots <= 3:
            raise SystemExit("Number of slots should be greater than 3")
        self.n_slots = n_slots
        self.slots = np.array([[] for i in range(n_slots)])  # or just [] or np.array[]

    def allocate_ads(self, ads):
        n_ads = len(ads)
        graph_matrix = np.zeros(shape=(n_ads, self.n_slots))

        for i in range(n_ads):
            for j in range(self.n_slots):
                b_i = ads[i].bid
                q_ij = ads[i].q[j]
                graph_matrix[i][j] = b_i * q_ij #perchè nella matrice è influent il bid?
        print("Ads allocating:")
        print(graph_matrix)
        print("Running hungarian...")
        res = hungarian_algorithm_max(cm(graph_matrix))
        m = res[1]
        edges = []
        for i in range(len(m)):
            for j in range(len(m[i])):
                if m[i][j] == 1:
                    edges.append([i, j])
        return edges
