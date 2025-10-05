import torch

# Create a 1D tensor
x = torch.tensor([1, 2, 3])
print(f"Original tensor: {x}")
print(f"Original shape: {x.shape}")

# Unsqueeze at dimension 0 (add a new leading dimension)
y = torch.unsqueeze(x, 0)
print(f"Unsqueezed tensor (dim 0): {y}")
print(f"Unsqueezed shape (dim 0): {y.shape}")

# Unsqueeze at dimension 1 (add a new dimension after the first)
z = torch.unsqueeze(x, 1)
print(f"Unsqueezed tensor (dim 1): {z}")
print(f"Unsqueezed shape (dim 1): {z.shape}")



print("==== Adjacent Matrix ====")


import scipy.sparse as sp
import numpy as np
from scipy.sparse import coo_matrix

n_users = 7
n_items = 4

A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
print(f"A = {A}")

row  = np.array([0, 3, 1, 0, 1])
col  = np.array([0, 3, 1, 2, 2])
data = np.array([4, 5, 7, 9, 1])
m = coo_matrix((data, (row, col)), shape=(4, 4))
m_t = m.transpose()
# Output:
# array([[4, 0, 9, 0],
#        [0, 7, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 5]])
print(f"m = {m.toarray()}")
print(f"m.row = {m.row}")
print(f"m.col = {m.col}")

print(f"Number of stored values (nnz): {m.nnz}")
print(f"Number of truly non-zero elements: {m.count_nonzero()}")


data_dict = dict(zip(zip(m.row, m.col + n_users), [1] * m.nnz))
print(f"*** 1. data_dict = {data_dict}")
for r, c in zip(m.row, m.col + n_users):
    print(r, c)

data_dict.update(dict(zip(zip(m_t.row + n_users, m_t.col), [1] * m_t.nnz)))
print(f"*** 2. data_dict = {data_dict}")
for r, c in zip(m_t.row + n_users, m_t.col):
    print(r, c)


for (row, col), value in data_dict.items():
    A[row, col] = value

print(f"A = {A}")


probability_matrix = torch.full([5, 10], 0.15)
print(probability_matrix)
print(probability_matrix.shape)

masked_indices = torch.bernoulli(probability_matrix).bool()
print("masked_indices:")
print(masked_indices)

indices_replaced = torch.bernoulli(torch.full([5, 10], 0.8)).bool() & masked_indices
print("indices_replaced:")
print(indices_replaced)
