"""Cap. 3: Determinante pela fórmula de Leibniz e suas propriedades."""
from itertools import permutations
from math import prod

import numpy as np


def sinal(s):
    inv = sum(1 for i in range(len(s)) for j in range(i + 1, len(s)) if s[i] > s[j])
    return (-1) ** inv


def det_leibniz(A):
    n = len(A)
    return sum(sinal(s) * prod(A[i, s[i]] for i in range(n)) for s in permutations(range(n)))


rng = np.random.default_rng(3)
A = np.round(rng.normal(size=(4, 4)), 2); B = np.round(rng.normal(size=(4, 4)), 2)
print(det_leibniz(A), np.linalg.det(A))
print(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))
print(np.linalg.det(A.T), np.linalg.det(A))
print(np.linalg.det(A[[2, 1, 0, 3]]))                    # troca de linhas: sinal oposto

V = np.array([[3, 1], [1, 2]]); print("área:", abs(np.linalg.det(V)))
X = np.array([[1, 0], [0, 2], [1, 1]]); print(np.sqrt(np.linalg.det(X.T @ X)))
