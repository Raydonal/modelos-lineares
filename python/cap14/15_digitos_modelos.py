"""Cap. 14: Dígitos: muitos classificadores, um só teste."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402
from comum.multiplas import bh  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
dig = pd.read_csv(raiz / "dados" / "digitos.csv")
P = dig.iloc[:, 1:].to_numpy(float) / 16; rot = dig.digito.to_numpy()
tr = np.arange(1200); te = np.arange(1200, len(P))  # 597 imagens
Xc = P - P[tr].mean(axis=0)
Xt = Xc[tr]; Y = (rot[tr, None] == np.arange(10)) * 1.0
w_, V_ = np.linalg.eigh(Xt.T @ Xt); ev = V_[:, ::-1]


def pca(k):
    return Xc @ ev[:, :k]


def centr(Sc):                                 # centróide euclidiano
    C = np.vstack([Sc[tr][rot[tr] == j].mean(axis=0)
                   for j in range(10)])
    d = ((Sc[te, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def lin(Sc, lam=0.0):                          # regressão linear
    St = Sc[tr]
    B = np.linalg.solve(St.T @ St + lam * np.eye(St.shape[1]),
                        St.T @ (Y - Y.mean(axis=0)))
    return (Sc[te] @ B).argmax(axis=1)


def knn(k):                                    # k vizinhos
    out = []
    for i in te:
        d = ((P[tr] - P[i]) ** 2).sum(axis=1)
        v = rot[tr][np.argsort(d, kind="stable")[:k]]
        out.append(np.bincount(v, minlength=10).argmax())
    return np.array(out)


C = {"cent": centr(Xc)}
for k in (10, 20, 40):
    C[f"cent{k}"] = centr(pca(k))
for k in (10, 20, 30, 40, 50):
    C[f"lin{k}"] = lin(pca(k))
for l in (0.01, 0.1, 1, 10, 100, 1000):
    C[f"ridge{l:g}"] = lin(Xc, l)
for k in (1, 3, 5, 7, 9):
    C[f"knn{k}"] = knn(k)
nomes = list(C)
Ac = np.column_stack([(C[k] == rot[te]) * 1.0 for k in nomes])
# Validação e teste final sorteados entre as 597 imagens
g = Gerador(597); o = np.argsort(g.unif(597))
val = o[:300]; fim = o[300:]; m = Ac.shape[1] - 1
acv = Ac[val].mean(axis=0); acf = Ac[fim].mean(axis=0)
b0 = nomes.index("lin30")
print(f"candidatos {m}; base lin30: val {acv[b0]:.4f}"
      f" fim {acf[b0]:.4f}")
for j in np.argsort(-acv, kind="stable"):
    print(f"{nomes[j]:<9} val {acv[j]:.4f} fim {acf[j]:.4f}")
# McNemar: d = acerto do candidato - acerto da base
D = np.delete(Ac[val], b0, axis=1) - Ac[val][:, [b0]]
vz = (D**2).sum(axis=0) > 0
nd = np.sqrt((D**2).sum(axis=0)); nd[nd == 0] = 1
tt = D.sum(axis=0) / nd
pv = 2 * stats.norm.cdf(-np.abs(tt))
g = Gerador(1998); N = 20000
E = (2.0 * (g.unif(N * 300) < 0.5) - 1).reshape(N, 300)
Tst = np.abs(E @ (D / nd))
cmax = np.sort(Tst.max(axis=1))[int(0.95 * N) - 1]
print(f"t min {tt.min():.2f} max {tt.max():.2f};"
      f" crítico max-t {cmax:.3f}"
      f" (Bonf {stats.norm.ppf(1 - 0.025 / m):.3f})")
za = stats.norm.ppf(0.975)
print(f"achados: sem ajuste {np.sum(np.abs(tt) > za)}"
      f" Bonf {np.sum(pv <= 0.05 / m)}"
      f" max-t {np.sum(np.abs(tt) > cmax)} BH {bh(pv, 0.05)}")
nd2 = [k for k in nomes if k != "lin30"]
print("sem ajuste: " + " ".join(n for n, v in zip(nd2, tt)
                                if abs(v) > za))
print("melhores que a base:"
      f" Bonf {np.sum((pv <= 0.05 / m) & (tt > 0))}"
      f" max-t {np.sum((np.abs(tt) > cmax) & (tt > 0))}"
      f" BH {np.sum((pv <= 0.05 * bh(pv, 0.05) / m) & (tt > 0))}")
Rm = np.corrcoef(D[:, vz].T)
print("correlação média entre as diferenças:"
      f" {Rm[np.triu_indices(Rm.shape[0], 1)].mean():.2f}")
# O vencedor: otimismo da validação
w = int(acv.argmax())
print(f"vencedor {nomes[w]}: val {acv[w]:.4f} fim {acf[w]:.4f}"
      f" (dif {acv[w] - acf[w]:.4f})")
print(f"todos: dif média val - fim {np.mean(acv - acf):.4f}")
# Sem nenhuma diferença real: o melhor de 20 em 300 imagens
g = Gerador(2015); pa = 0.96; ep = np.sqrt(pa * (1 - pa) / 300)
z = g.matriz_normal(20000, 20).max(axis=1)
print(f"máx. de 20 nulos independentes: +{z.mean() * ep:.4f}"
      f" ({z.mean():.2f} erros-padrão)")
