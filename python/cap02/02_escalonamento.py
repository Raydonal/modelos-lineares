"""Cap. 2: Forma escalonada reduzida, bases dos quatro subespaços e fatoração de posto."""
import numpy as np


def escalonar(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivos, i = [], 0
    for j in range(n):
        if i >= m:
            break
        k = i + np.argmax(np.abs(R[i:, j]))     # pivotamento parcial
        if abs(R[k, j]) < tol:
            continue                            # coluna sem pivô
        R[[i, k]] = R[[k, i]]                   # troca de linhas
        R[i] /= R[i, j]                         # pivô igual a 1
        for l in range(m):
            if l != i:
                R[l] -= R[l, j] * R[i]
        pivos.append(j)
        i += 1
    R[np.abs(R) < tol] = 0
    return R, pivos


A = np.array([[1, 2, 0, 3], [2, 4, 1, 7], [3, 6, 1, 10]], dtype=float)
R, pivos = escalonar(A)
print(R, pivos)
r = len(pivos)

base_col, base_lin = A[:, pivos], R[:r]
livres = [j for j in range(A.shape[1]) if j not in pivos]
base_nul = np.zeros((A.shape[1], len(livres)))
for c, f in enumerate(livres):
    base_nul[f, c] = 1
    base_nul[pivos, c] = -R[:r, f]
print(base_nul, A @ base_nul, sep="\n")

print(np.abs(base_col @ base_lin - A).max())     # fatoração de posto

for b in ([1, 3, 4], [1, 3, 5]):                 # Rouché-Capelli
    print(r, np.linalg.matrix_rank(np.column_stack([A, b])))
