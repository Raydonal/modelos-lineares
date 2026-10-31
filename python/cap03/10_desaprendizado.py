"""Cap. 3 -- Desaprendizado de máquina por Woodbury."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
db = pd.read_csv(raiz / "dados" / "diabetes.csv")
X = np.column_stack([np.ones(len(db)), db.iloc[:, :10].to_numpy()])
y = db["y"].to_numpy(float)
n, p = X.shape

# O modelo guardado: X'X, sua inversa G, o estimador e os resíduos
A = X.T @ X; G = np.linalg.inv(A); beta = G @ X.T @ y
e = y - X @ beta; h = np.sum((X @ G) * X, axis=1)


def esquecer(S):
    """beta_(S) = beta - G X_S' (I - H_SS)^{-1} e_S."""
    XS = X[S]
    K = np.eye(len(S)) - XS @ G @ XS.T     # I - H_SS
    return beta - G @ XS.T @ np.linalg.solve(K, e[S])


def retreinar(S):
    Xr, yr = np.delete(X, S, 0), np.delete(y, S)
    return np.linalg.lstsq(Xr, yr, rcond=None)[0]


g = Gerador(3)
S_acaso = np.sort(np.argsort(g.unif(n))[:20])  # 20 pedidos ao acaso
S_alav = np.argsort(-h, kind="stable")[:20]    # 20 de maior alavanca
for nome, S in {"acaso": S_acaso, "alavanca": S_alav}.items():
    bS = esquecer(S)
    dif = np.abs(bS - retreinar(S)).max()
    mud = np.sqrt(np.mean((X @ (beta - bS))**2))
    print(f"{nome}: dif. retreino = {dif:.1e},"
          f" h médio = {h[S].mean():.4f},"
          f" mudança no ajuste = {mud:.2f}")
    print(f"  |resíduo| médio antes = {np.abs(e[S]).mean():.2f},"
          f" depois = {np.abs(y[S] - X[S] @ bS).mean():.2f}")

# Pedidos um a um: 20 atualizações de posto um (Sherman-Morrison)
Gs = G.copy(); bs = X.T @ y
for i in S_acaso:
    x = X[i]; Gx = Gs @ x
    Gs += np.outer(Gx, Gx) / (1 - x @ Gx)
    bs -= x * y[i]
Xr = np.delete(X, S_acaso, 0); G_ret = np.linalg.inv(Xr.T @ Xr)
di = np.abs(Gs - G_ret).max() / np.abs(Gs).max()
db_ = np.abs(Gs @ bs - retreinar(S_acaso)).max()
print(f"sequencial: dif. relativa da inversa = {di:.1e},"
      f" dif. beta = {db_:.1e}")

# O vazamento: d = A (beta - beta_(i)) = x_i e_(i)
i = S_acaso[0]; bi = esquecer([i])
d = A @ (beta - bi)
x_rec = d / d[0]                       # o intercepto fixa a escala
y_rec = x_rec @ bi + d[0]              # y_i = x_i' beta_(i) + e_(i)
ex, ey = np.abs(x_rec - X[i]).max(), abs(y_rec - y[i])
print(f"paciente {i + 1}: erro em x = {ex:.1e}, erro em y = {ey:.1e}")

# Custo num problema grande (n = 200 000, p = 50, k = 20)
rng = np.random.default_rng(1); N = 200_000
Xg = np.column_stack([np.ones(N), rng.standard_normal((N, 49))])
yg = Xg @ rng.standard_normal(50) + rng.standard_normal(N)
Gg = np.linalg.inv(Xg.T @ Xg); bg = Gg @ Xg.T @ yg
XS = Xg[:20]
t0 = time.perf_counter()
np.linalg.lstsq(Xg[20:], yg[20:], rcond=None)
t_ret = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(100):
    K = np.eye(20) - XS @ Gg @ XS.T
    bg - Gg @ XS.T @ np.linalg.solve(K, yg[:20] - XS @ bg)
t_upd = (time.perf_counter() - t0) / 100
print(f"retreinar {t_ret:.3f}s, atualizar {t_upd:.1e}s,"
      f" razão {t_ret / t_upd:.0f}")
