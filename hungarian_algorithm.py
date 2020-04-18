import numpy as np
from scipy.optimize import linear_sum_assignment


# The Hungarian algorithm returns minimal cost matching
# Key idea of algorithm:
#   If number is added to all of the entries of anyone row or column of matrix,
#   then an optimal assignment for the resulting cost matrix
#   is also an optimal assignment for the original cost matrix
#
# So we can compute the maximum matching by minimizing the loss instead of maximizing the initial weights
# We generate new matrix by subtracting from the maximum entry the values of other entries
#
# Otherwise, If the original graph is not balanced we can dummy variables
# with the maximum cost (zero as we are maximizing) as value

# Algorithms steps
# 1. Subtract the smallest entry in each row from all the other entries in the row
# 2. Subtract the smallest entry in each column from all the other entries in the column
# 3. Draw lines through the row and columns that have the 0 entries such that the fewest lines possible are drawn
# 4. If there are number of lines are drawn are equals to number N of rows and columns,
#    an optimal assignment of zeros is possible and the algorithm is finished.
# 5. Find the smallest entry not covered by any line. Subtract this entry from each row that is not crossed out,
#    and then add it to each column that is crossed out. Go to the step 3.


# Step 1. Subtract the smallest entry in each row from all the other entries in the row
def step1(m):
    for i in range(m.shape[0]):
        m[i, :] = m[i, :] - np.min(m[i, :])


# Step 2. Subtract the smallest entry in each column from all the other entries in the column
def step2(m):
    for i in range(m.shape[1]):
        m[:, i] = m[:, i] - np.min(m[:, i])


# Step 3. Draw lines through the row and columns that have the 0 entries such that the fewest lines possible are drawn
# We have to find minimum number of lines, rows and columns, which have the 0 entries
def step3(m):
    # Step 3.1. Find a start assignment covering as many tasks as possible
    # Step 3.2. Mark all rows having no assignment
    # Step 3.3. Mark all unmarked columns having zeros in newly marked row(s)
    # Step 3.4. Mark all rows having assignments in newly marked columns
    # Step 3.5. Repeat this procedure for all not assignment rows
    # Step 3.6. Select marked columns and unmarked rows

    dim = m.shape[0]  # value of number of rows or columns
    assigned = np.array([])  # list of indexes of assigned rows
    # Zeros metrics in which we store assignments.
    # The entry ij of the matrix will have value 1 if the corresponding row i is assigned to column j
    assignments = np.zeros(m.shape, dtype=int)

    for i in range(0, dim):
        for j in range(0, dim):
            # if we found zero, and rows and columns are not assigned
            if m[i, j] == 0 and np.sum(assignments[:, j]) == 0 and np.sum(assignments[i, :]) == 0:
                assignments[i, j] = 1  # assign row to the column
                assigned = np.append(assigned, i)  # indexes of the rows

    # use assigned array to initialize the marked rows variable
    rows = np.linspace(0, dim - 1, dim).astype(int)
    marked_rows = np.setdiff1d(rows, assigned)  # non assigned rows
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])

    # iterate until we have new marked rows
    while len(new_marked_rows) > 0:
        new_marked_cols = np.array([], dtype=int)
        # mark columns that have 0 at that row
        for nr in new_marked_rows:
            zeros_cols = np.argwhere(m[nr, :] == 0).reshape(-1)  # indexes of the columns
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols))
        marked_cols = np.append(marked_cols, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)

        # iterate over new marked cols
        for nc in new_marked_cols:
            # update value of new marked rows by appending indexes of the rows that as an assignment in that column
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:, nc] == 1).reshape(-1))

        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))

    # return indexes of unmarked rows and indexes of marked columns
    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)


def step5(m, covered_rows, covered_cols):
    # modify matrix by finding the minimum entry in covered cells
    # and subtracting this value from each row that is not cross out
    # then add it to each column that it cross out

    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0] - 1, m.shape[0] - 1, m.shape[0]), covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1] - 1, m.shape[1] - 1, m.shape[1]), covered_cols).astype(int)

    min_val = np.max(m)
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if m[i, j] < min_val:
                min_val = m[i, j]

    for i in uncovered_rows.astype(int):
        m[i, :] -= min_val

    for j in covered_cols.astype(int):
        m[:, j] += min_val

    return m


def find_rows_single_zero(matrix):
    for i in range(0, matrix.shape[0]):
        if np.sum(matrix[i, :] == 0) == 1:
            j = np.argwhere(matrix[i, :] == 0).reshape(-1)[0]
            return i, j
    return False


def find_cols_single_zero(matrix):
    for i in range(0, matrix.shape[1]):
        if np.sum(matrix[:, i] == 0) == 1:
            j = np.argwhere(matrix[:, i] == 0).reshape(-1)[0]
            return i, j
        return False


def assignment_single_zero_lines(m, assignment):
    # assignment - boolean matrix
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


def first_zero(m):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]


# Final assignment
# 1. Examine the rows successively until a row-wise exactly single zero is found
#    mark this zero by 1 to make the assignment
# 2. Mark all zeros lying in the column of the marked zero
# 3. Do the same procedure for the columns also
# 4. Repeat until there are no rows and columns with single zeros
# 5. If there lies more than one of the unmarked zeroes in any column or row,
#    then mark '1' one of the unmarked zeroes arbitrarily
#    and mark a cross in the cells of remaining zeroes in its row and column
# 6. Repeat all the process until no unmarked zero is left in the matrix
def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assignment_single_zero_lines(m, assignment)

    while np.sum(m == 0) > 0:
        i, j = first_zero(m)
        assignment[i, j] = 1
        m[i, :] += 1
        m[:, j] += 1
        assignment = assignment_single_zero_lines(m, assignment)

    return assignment * initial_matrix, assignment


def hungarian_algorithm(matrix):
    m = matrix.copy()
    step1(m)
    step2(m)
    n_lines = 0
    max_length = np.maximum(m.shape[0], m.shape[1])
    while n_lines != max_length:
        lines = step3(m)
        n_lines = len(lines[0] + len(lines[1]))

        if n_lines != max_length:
            step5(m, lines[0], lines[1])

    return final_assignment(matrix, m)


a = np.random.randint(100, size=(3, 3))

res = hungarian_algorithm(a)

print("\n Optimal matching: \n ", res[1], "\n Value: ", np.sum(res[0]))
