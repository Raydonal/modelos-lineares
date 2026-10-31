"""Cap. 3: Estática comparativa no modelo IS-LM pela regra de Cramer."""
import numpy as np

p = dict(c0=100, c1=0.75, i0=1050, b=50, k=0.25, h=100, T=200, G=250, MP=500)


def sistema(p):
    A = np.array([[1 - p["c1"], p["b"]], [p["k"], -p["h"]]])
    d = np.array([p["c0"] - p["c1"] * p["T"] + p["i0"] + p["G"], p["MP"]])
    return A, d


def cramer(A, d):
    out = []
    for j in range(2):
        Aj = A.copy(); Aj[:, j] = d
        out.append(np.linalg.det(Aj) / np.linalg.det(A))
    return np.array(out)


A, d = sistema(p)
print("det =", np.linalg.det(A))
print("equilíbrio (Y, r):", cramer(A, d))
print("dG:", cramer(A, [1, 0]), " dM/P:", cramer(A, [0, 1]))
