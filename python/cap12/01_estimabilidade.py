"""Cap. 12: Funções estimáveis num fator com três níveis"""
import numpy as np

y = np.array([5, 7, 10, 12, 9.]); grupo = np.array([1, 1, 2, 2, 3])
X = np.column_stack([np.ones(5), grupo[:, None] == [1, 2, 3]])
XtX = X.T @ X; Xty = X.T @ y
G1 = np.linalg.pinv(XtX, rcond=1.5e-8)        # Moore-Penrose
G2 = np.zeros((4, 4))                         # zera o intercepto
G2[1:, 1:] = np.linalg.inv(XtX[1:, 1:])
b1 = G1 @ Xty; b2 = G2 @ Xty
print("b_mp ", b1.round(3), "\nb_ref", b2.round(3))  # mu, a1, a2, a3
L = {"mu+a1": [1, 1, 0, 0], "a1-a2": [0, 1, -1, 0],
     "a1-(a2+a3)/2": [0, 1, -.5, -.5],
     "mu": [1, 0, 0, 0], "a1": [0, 1, 0, 0]}
# teste: l'(I - G X'X) = 0 <=> l no espaço linha de X
print(f"{'':13}{'estim.':>7}{'mp':>6}{'ref':>6}"
      f"{'v_mp':>7}{'v_ref':>7}")
for nome, l in L.items():
    l = np.array(l, float)
    e = np.max(np.abs(l @ (np.eye(4) - G1 @ XtX))) < 1e-10
    print(f"{nome:13}{e:>7}{l @ b1:6.1f}{l @ b2:6.1f}"
          f"{l @ G1 @ l:7.3f}{l @ G2 @ l:7.3f}")
