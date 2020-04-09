import numpy as np
from scipy.optimize import linear_sum_assignment

#-----------------------------------------STEP 1------------------------------------------
# For every row subtract the smallest entry of the row to all the elements of the row
def step1(m):
    for i in range(m.shape[0]):#ritorna le  righe della matrice
        m[i, :] = m[i, :] - np.min(m[i, :])

#-----------------------------------------STEP 2------------------------------------------
# For every col subtract the smallest entry of the row to all the elements of the col
def step2(m):
    for i in range(m.shape[1]):#ritorna le  colonne della matrice
        m[:, i] = m[:, i] - np.min(m[:, i])

#-----------------------------------------STEP 3------------------------------------------
def step3(m):
    #Draw lines through the row and column that have 0 entries such that the fewest line possible are drawn
    #Con le righe copro gli 0

    dim = m.shape[0] #ritorna le  righe della matrice
    assigned = np.array([]) #inizializziamo una lista
    assignments = np.zeros(m.shape, dtype=int) #inizializziamo una mantrice

    # 1. Find a start assignement overing as many task (Y) as possible
    for i in range(dim):
        for j in range(dim):
            if m[i, j] == 0 and np.sum(assignments[:, j]) == 0 and np.sum(assignments[i, :]) == 0:
                assignments[i, j] = 1 #we set the assignement
                assigned = np.append(assigned, i) #we store in the array the value of the index of the array

    rows = np.linspace(0, dim - 1, dim).astype(int) # 2. Mark all rows having no assignement
    marked_rows = np.setdiff1d(rows, assigned)
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])

    while len(new_marked_rows) > 0:
        new_marked_cols = np.array([], dtype=int)  # 3. Mark all unmarked column having zero in newly market row
        for nr in new_marked_rows:
            zeros_cols = np.argwhere(m[nr, :] == 0).reshape(-1)
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))
        marked_cols = np.append(marked_cols, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)

        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:, nc] == 1).reshape(-1)) # 4. Mark all rows having assignements in newly marked column
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))
    # 5. Repeat
    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)# 6.Select market columns and unmarked rows.

#---------------------------------------??STEP 4??------------------------------------------
#If there are N lines drawn an optimal assignement of zeros is possible and the alhorithm is finished. Otherwise, go to the next step.
#Where N is the numer of column/rows

#-----------------------------------------STEP 5------------------------------------------
#Find hte smallest entry not covered by any line. Subtract this entry from each row that is not crossed out, and then
#add it to each coulumn that IS crossed out

def step5(m, covered_rows, covered_cols):
    #let's find uncovered rows and coulmn
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0] - 1, m.shape[0]), covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1] - 1, m.shape[1]), covered_cols).astype(int)
    min_val = np.max(m) #????????

    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if m[i, j] < min_val:  #iteriamo sulla matrice e troviamo il valore minimo
                min_val = m[i, j]

    for i in uncovered_rows.astype(int):
        m[i, :] -= min_val

    for j in covered_cols.astype(int):
        m[:, j] += min_val

    return m



#-----------------------------------------FINAL ASSIGNEMENT------------------------------------------
#1.Examine the rows, until it's founf a row where there is a single zero, than mark this zero by "1" to make the assignamente
#2.Mark all the zeros lying in the column of hte marked zero
#3.Mark all the zeros lying in the row of hte marked zero
#4.Repeat until there are no nows and column with single zeros
#5.If we have one row/column with more than one market 0 ("1"), we have to keep one and discard the other
#6.Repeat until no unmarked zero is left in the matrix



def find_rows_single_zero(m):
    for i in range(m.shape[0]):
        if np.sum(m[i, :] == 0) == 1:
            j = np.argwhere(m[i, :] == 0).reshape(-1)[0]
            return i, j
    return False


def find_cols_single_zero(m):
    for j in range(m.shape[1]):
        if np.sum(m[:, j] == 0) == 1:
            i = np.argwhere(m[:, j] == 0).reshape(-1)[0]
            return i, j
    return False


def assignment_single_zero_lines(m, assignment):
    val = find_rows_single_zero(m)
    while val:
        i, j = val[0], val[1]
        m[i, j] += 1
        m[:, j] += 1
        assignment[i, j] = 1
        val = find_rows_single_zero(m)

    val = find_cols_single_zero(m)
    while val:
        i, j = val[0], val[1]
        m[i, :] += 1
        m[i, j] += 1
        assignment[i, j] = 1
        val = find_cols_single_zero(m)
    return assignment

def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assignment_single_zero_lines(m, assignment)
    while np.sum(m == 0) > 0:
        i, j = first_zero(m)
        assignment[i, j] = 1
        m[i, :] += 1
        m[:, j] += 1
        assignment = assignment_single_zero_lines(m, assignment) #5

    return assignment * initial_matrix, assignment


# This implementation aims at minimizing cost expressed in the adj matrix
def hungarian_algorithm(matrix):
    m = matrix.copy()
    step1(m)
    step2(m)
    n_lines = 0
    max_length = np.maximum(m.shape[0], m.shape[1])
    while n_lines != max_length:
        lines = step3(m)
        n_lines = len(lines[0]) + len(lines[1])
        if n_lines != max_length:
            step5(m, lines[0], lines[1])
    return final_assignment(matrix, m)

def first_zero(m):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]


for i in range(10000):
    a = np.random.randint(100, size=(3, 3))
    res = hungarian_algorithm(a)
    print("\nMatrix:\n", a, "\nOptimal Matching:\n", res[1], "\nValue: ", np.sum(res[0]))