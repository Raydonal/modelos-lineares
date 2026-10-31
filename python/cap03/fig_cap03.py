"""Cap. 3 -- Figuras de dados do capítulo (geradas em figuras/cap03/)."""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap03"
LARG = 5.3


# ---------------------------------------------------------------------------
# 1. IS-LM: equilíbrio e política fiscal
# ---------------------------------------------------------------------------
def fig_islm():
    p = dict(c0=100, c1=0.75, i0=1050, b=50, k=0.25, h=100, T=200, G=250, MP=500)
    Y = np.linspace(3200, 4800, 100)
    r_is = lambda G: (p["c0"] - p["c1"] * p["T"] + p["i0"] + G - (1 - p["c1"]) * Y) / p["b"]
    r_lm = (p["k"] * Y - p["MP"]) / p["h"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(Y, r_is(250), color=CATEG[0], label="IS ($G=250$)")
    ax.plot(Y, r_is(400), color=CATEG[0], ls="--", label="IS ($G=400$)")
    ax.plot(Y, r_lm, color=CATEG[1], label="LM")
    for G, Ye in ((250, 4000), (400, 4400)):
        re = (p["k"] * Ye - p["MP"]) / p["h"]
        ax.plot(Ye, re, "o", color="#222", ms=4)
    ax.annotate("", xy=(4400, 6), xytext=(4000, 5), arrowprops=dict(arrowstyle="->", color=COR["cinza"]))
    ax.set_xlabel("renda $Y$"); ax.set_ylabel("juros $r$ (%)")
    ax.set_ylim(2, 9); ax.legend(fontsize=7, loc="upper right")
    salvar(fig, CAP, "islm")


# ---------------------------------------------------------------------------
# 2. Correlação espacial no experimento de aveia
# ---------------------------------------------------------------------------
def fig_espacial():
    a = pd.read_csv(DADOS / "yates_oats.csv")
    y = a["yield"].to_numpy(float)
    dm = lambda s: pd.get_dummies(s, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(a)), dm(a["block"]), dm(a["nitro"]), dm(a["gen"])])
    coord = np.column_stack([(a["col"] - 1) * 44 * 0.2012, (a["row"] - 1) * 28.4 * 0.2012])
    D = cdist(coord, coord); n = len(y)
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    def logvero(phi):
        R = np.eye(n) if phi == 0 else np.exp(-D / phi)
        RiX, Riy = np.linalg.solve(R, X), np.linalg.solve(R, y)
        beta = np.linalg.solve(X.T @ RiX, X.T @ Riy)
        r = y - X @ beta
        s2 = r @ np.linalg.solve(R, r) / n
        return -n / 2 * np.log(2 * np.pi * s2) - np.linalg.slogdet(R)[1] / 2 - n / 2

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 3.0), gridspec_kw={"width_ratios": [0.8, 1.4]})
    lim = np.abs(res).max()
    sc = a1.scatter(coord[:, 0], coord[:, 1], c=res, cmap="RdBu_r", vmin=-lim, vmax=lim,
                    s=60, marker="s", edgecolors="#555", linewidths=0.3)
    a1.set_aspect("equal"); a1.set_xlabel("m"); a1.set_ylabel("m"); a1.grid(False)
    a1.set_title("(a) resíduos no campo", fontsize=8.5, loc="left")
    cb = fig.colorbar(sc, ax=a1, shrink=0.8); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    phis = np.concatenate([[0], np.linspace(0.5, 80, 80)])
    ll = np.array([logvero(f) for f in phis])
    a2.plot(phis[1:], ll[1:], color=CATEG[0])
    a2.axhline(ll[0], color=CATEG[1], ls="--", lw=1)
    a2.text(55, ll[0] + 0.25, "independência ($\\phi=0$)", fontsize=7, color=CATEG[1], ha="center")
    i = np.argmax(ll); a2.plot(phis[i], ll[i], "o", color="#222", ms=4)
    a2.annotate(f"$\\hat\\phi\\approx{phis[i]:.0f}$ m", (phis[i], ll[i]), (phis[i] + 8, ll[i] - 1.5), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=COR["cinza"], lw=0.6))
    a2.set_xlabel(r"alcance $\phi$ (m)"); a2.set_ylabel("log-verossimilhança perfilada")
    a2.set_title("(b) perfil em $\\phi$", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "espacial_aveia")


# ---------------------------------------------------------------------------
# 3. LOO e mínimos quadrados recursivos (diabetes)
# ---------------------------------------------------------------------------
def fig_loo_rls():
    db = pd.read_csv(DADOS / "diabetes.csv")
    X = np.column_stack([np.ones(len(db)), db.iloc[:, :10].to_numpy()])
    y = db["y"].to_numpy(float); n, p = X.shape
    Q, R = np.linalg.qr(X); b_mqo = np.linalg.solve(R, Q.T @ y)
    e = y - X @ b_mqo; h = np.sum(Q**2, 1); loo = e / (1 - h)
    m0 = 20
    P = np.linalg.inv(X[:m0].T @ X[:m0]); b = P @ X[:m0].T @ y[:m0]
    hist = [b.copy()]
    for t in range(m0, n):
        x = X[t]; Px = P @ x
        P -= np.outer(Px, Px) / (1 + x @ Px); b = b + P @ x * (y[t] - x @ b); hist.append(b.copy())
    hist = np.array(hist)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.vlines(np.arange(n), 0, h, color=CATEG[0], lw=0.6)
    a1.axhline(p / n, color=COR["cinza"], ls="--", lw=0.9)
    a1.axhline(2 * p / n, color=CATEG[1], ls=":", lw=0.9)
    a1.text(n * 0.99, p / n * 0.55, "$p/n$", fontsize=7, ha="right", color=COR["cinza"])
    a1.text(n * 0.99, 2 * p / n * 1.08, "$2p/n$", fontsize=7, ha="right", color=CATEG[1])
    a1.set_xlabel("observação $i$"); a1.set_ylabel("alavanca $h_{ii}$")
    a1.set_title(r"(a) $e_{(i)}=e_i/(1-h_{ii})$: alavancas", fontsize=8.5, loc="left")
    tt = np.arange(m0, n + 1)
    for j, (nome, cor) in enumerate(zip(["bmi", "bp", "s5"], CATEG)):
        k = list(db.columns).index(nome) + 1
        sd = X[:, k].std()
        a2.plot(tt, hist[:, k] * sd, color=cor, lw=1.3, label=nome)
        a2.axhline(b_mqo[k] * sd, color=cor, ls=":", lw=0.9)
    a2.set_xlabel("número de observações processadas"); a2.set_ylabel("coeficiente × desvio-padrão")
    a2.set_ylim(-20, 80)
    a2.legend(fontsize=7, loc="upper right")
    a2.set_title("(b) mínimos quadrados recursivos", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "loo_rls")


# ---------------------------------------------------------------------------
# 4. Fluxo normalizante: densidade "banana" pela mudança de variáveis
# ---------------------------------------------------------------------------
def fig_fluxo():
    s_fun = lambda x1: np.log(0.5) + 0.3 * x1
    t_fun = lambda x1: 0.5 * x1**2 - 1
    rng = np.random.default_rng(4)
    z = rng.normal(size=(1500, 2))
    x = np.column_stack([z[:, 0], t_fun(z[:, 0]) + np.exp(s_fun(z[:, 0])) * z[:, 1]])
    g1, g2 = np.linspace(-3.5, 3.5, 300), np.linspace(-2.5, 6, 300)
    G1, G2 = np.meshgrid(g1, g2)
    Z2 = (G2 - t_fun(G1)) / np.exp(s_fun(G1))
    dens = norm.pdf(G1) * norm.pdf(Z2) / np.exp(s_fun(G1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.scatter(z[:, 0], z[:, 1], s=3, alpha=0.4, color=CATEG[0], lw=0)
    a1.set_title(r"(a) base: $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$", fontsize=8.5, loc="left")
    a1.set_xlim(-3.5, 3.5); a1.set_ylim(-3.5, 3.5); a1.set_aspect("equal")
    a2.scatter(x[:, 0], x[:, 1], s=3, alpha=0.25, color=CATEG[0], lw=0)
    a2.contour(G1, G2, dens, levels=6, colors=CATEG[1], linewidths=0.9)
    a2.set_xlim(-3.5, 3.5); a2.set_ylim(-2.5, 6)
    a2.set_title(r"(b) $p(\mathbf{x})=\varphi(\mathbf{z}(\mathbf{x}))\,|\det J|$", fontsize=8.5, loc="left")
    for ax in (a1, a2):
        ax.set_xlabel("1ª coord."); ax.set_ylabel("2ª coord.")
    fig.tight_layout()
    salvar(fig, CAP, "fluxo")


# ---------------------------------------------------------------------------
# 5. Woodbury: tempo de resolver (A0 + U U')x = b
# ---------------------------------------------------------------------------
def fig_woodbury():
    rng = np.random.default_rng(0)
    ns = [250, 500, 1000, 2000, 3000]
    k = 5; td, tw = [], []
    for n in ns:
        a0 = rng.uniform(1, 2, n); U = rng.normal(size=(n, k)); b = rng.normal(size=n)
        M = np.diag(a0) + U @ U.T
        t0 = time.perf_counter()
        for _ in range(3):
            np.linalg.solve(M, b)
        td.append((time.perf_counter() - t0) / 3)
        t0 = time.perf_counter()
        for _ in range(3):
            AU, Ab = U / a0[:, None], b / a0
            Ab - AU @ np.linalg.solve(np.eye(k) + U.T @ AU, U.T @ Ab)
        tw.append((time.perf_counter() - t0) / 3)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.loglog(ns, td, "o-", color=CATEG[1], ms=4, label="direto (LU), $O(n^3)$")
    ax.loglog(ns, tw, "o-", color=CATEG[0], ms=4, label="Woodbury, $O(nk^2)$")
    ax.set_xticks(ns, [str(v) for v in ns]); ax.minorticks_off()
    ax.set_xlabel("$n$"); ax.set_ylabel("segundos"); ax.legend(fontsize=7)
    salvar(fig, CAP, "woodbury_tempo")


if __name__ == "__main__":
    fig_islm()
    fig_espacial()
    fig_loo_rls()
    fig_fluxo()
    fig_woodbury()
