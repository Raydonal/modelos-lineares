"""Cap. 4: Figuras de dados do capítulo (geradas em figuras/cap04/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap04"
LARG = 5.3


# ---------------------------------------------------------------------------
# 1. Aveia: soluções diferentes, funções estimáveis iguais
# ---------------------------------------------------------------------------
def fig_aveia():
    a = pd.read_csv(DADOS / "yates_oats.csv")
    ind = lambda s: pd.get_dummies(s).to_numpy(float)
    X = np.column_stack([np.ones(len(a)), ind(a["block"]), ind(a["nitro"]), ind(a["gen"])])
    nomes = [r"$\mu$"] + [f"$\\beta_{i}$" for i in range(1, 7)] + \
            [f"$\\nu_{{{d}}}$" for d in ("0", "0,2", "0,4", "0,6")] + [r"$\gamma_{GR}$", r"$\gamma_{Ma}$", r"$\gamma_{Vi}$"]
    y = a["yield"].to_numpy(float)
    XtX, Xty = X.T @ X, X.T @ y
    e = lambda js: np.isin(np.arange(14), js).astype(float)
    K = {"tratamento": [e([1]), e([7]), e([11])],
         "soma zero": [e(range(1, 7)), e(range(7, 11)), e(range(11, 14))],
         "Moore–Penrose": [e([0]) - e(range(1, 7)), e([0]) - e(range(7, 11)), e([0]) - e(range(11, 14))]}
    sol = {k: np.linalg.solve(XtX + np.array(v).T @ np.array(v), Xty) for k, v in K.items()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    pos = np.arange(14)
    for i, (k, b) in enumerate(sol.items()):
        a1.plot(pos + (i - 1) * 0.22, b, "o", ms=3.6, color=CATEG[i], label=k)
    a1.set_xticks(pos, nomes, fontsize=6.5, rotation=60)
    a1.axhline(0, color=COR["cinza"], lw=0.5)
    a1.set_ylabel("estimativa (1/4 lb por subparcela)")
    a1.set_title("(a) três soluções das equações normais", fontsize=8.5, loc="left")
    a1.legend(fontsize=6.5, loc="upper right", handletextpad=0.2)

    # funções estimáveis: diferenças de doses e de variedades
    L = {r"$\nu_{0,2}-\nu_0$": e([8]) - e([7]), r"$\nu_{0,4}-\nu_0$": e([9]) - e([7]),
         r"$\nu_{0,6}-\nu_0$": e([10]) - e([7]), r"$\gamma_{Ma}-\gamma_{GR}$": e([12]) - e([11]),
         r"$\gamma_{Vi}-\gamma_{GR}$": e([13]) - e([11])}
    for i, (k, b) in enumerate(sol.items()):
        v = [l @ b for l in L.values()]
        a2.plot(v, np.arange(len(L)) + (i - 1) * 0.0, "o", ms=[7, 5, 3][i], color=CATEG[i],
                markerfacecolor="none" if i < 2 else CATEG[i], mew=1.1)
    a2.set_yticks(np.arange(len(L)), list(L.keys()), fontsize=7)
    a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.set_xlabel("estimativa")
    a2.set_title("(b) funções estimáveis", fontsize=8.5, loc="left")
    a2.invert_yaxis()
    fig.tight_layout()
    salvar(fig, CAP, "aveia_restricoes")


# ---------------------------------------------------------------------------
# 2. NFL 1981: grafo de jogos após duas semanas e ratings da temporada
# ---------------------------------------------------------------------------
def fig_massey():
    nfl = pd.read_csv(DADOS / "nfl.csv")
    s = nfl[nfl["ano"] == 1981]
    times = sorted(set(s["mandante"]) | set(s["visitante"]))
    idx = {t: i for i, t in enumerate(times)}

    def delineamento(j):
        X = np.zeros((len(j), 29)); X[:, -1] = 1
        X[np.arange(len(j)), j["mandante"].map(idx)] = 1
        X[np.arange(len(j)), j["visitante"].map(idx)] = -1
        return X

    # componentes conexas após a semana 2
    j2 = s[s["semana"] <= 2]
    rot = np.arange(28)
    while True:
        novo = rot.copy()
        for m, v in zip(j2["mandante"].map(idx), j2["visitante"].map(idx)):
            novo[[m, v]] = novo[[m, v]].min()
        if np.array_equal(novo, rot):
            break
        rot = novo
    grupos = {g: i for i, g in enumerate(np.unique(rot))}
    comp = np.array([grupos[g] for g in rot])
    ordem = np.lexsort((np.arange(28), comp))
    ang = np.empty(28); ang[ordem] = np.linspace(0, 2 * np.pi, 28, endpoint=False) + np.pi / 2
    xy = np.column_stack([np.cos(ang), np.sin(ang)])

    X = delineamento(s); y = (s["pontos_mandante"] - s["pontos_visitante"]).to_numpy(float)
    b = np.linalg.pinv(X) @ y
    r = b[:28]; o = np.argsort(r)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 3.3), gridspec_kw={"width_ratios": [1, 1.05]})
    for m, v in zip(j2["mandante"].map(idx), j2["visitante"].map(idx)):
        a1.plot(*xy[[m, v]].T, color=CATEG[comp[m]], lw=0.8, alpha=0.8, zorder=1)
    a1.scatter(*xy.T, s=22, c=[CATEG[c] for c in comp], edgecolors="white", linewidths=0.8, zorder=2)
    for k, t in enumerate(times):
        a1.text(*(1.15 * xy[k]), t, fontsize=5.3, ha="center", va="center", color="#333")
    a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("(a) semanas 1 e 2: duas componentes", fontsize=8.5, loc="left", pad=10)

    a2.hlines(np.arange(28), 0, r[o], color=COR["cinza"], lw=0.6)
    a2.plot(r[o], np.arange(28), "o", ms=3.4, color=CATEG[0])
    a2.set_yticks(np.arange(28), [times[k] for k in o], fontsize=5.3)
    a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.set_xlabel("rating (pontos acima da média)")
    a2.set_title(f"(b) ratings de 1981 (mando = {b[-1]:.1f} pontos)".replace(".", ","), fontsize=8.5, loc="left")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    salvar(fig, CAP, "massey_nfl")


# ---------------------------------------------------------------------------
# 3. Dupla descida
# ---------------------------------------------------------------------------
def fig_dupla_descida():
    rng = np.random.default_rng(2026)
    D, n, sigma = 500, 50, 0.5
    beta = np.full(D, 2 / np.sqrt(D))
    ps = np.unique(np.r_[np.arange(2, 46, 2), np.arange(46, 56), np.arange(56, 100, 4), np.arange(100, 501, 20)])
    R = np.empty((200, len(ps))); N = np.empty_like(R)
    for k in range(200):
        X = rng.normal(size=(n, D)); y = X @ beta + sigma * rng.normal(size=n)
        for j, p in enumerate(ps):
            b = np.linalg.pinv(X[:, :p]) @ y
            R[k, j] = np.sum((beta[:p] - b) ** 2) + np.sum(beta[p:] ** 2) + sigma**2
            N[k, j] = np.linalg.norm(b)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    for ax, M, rot in ((a1, R, "risco de previsão"), (a2, N, r"norma $\|\widehat{\mathbf{b}}\|$")):
        q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
        ax.fill_between(ps, q1, q3, color=CATEG[0], alpha=0.18, lw=0)
        ax.plot(ps, med, color=CATEG[0], lw=1.4)
        ax.axvline(n, color=COR["cinza"], ls=":", lw=0.9)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel("número de covariáveis $p$"); ax.set_ylabel(rot)
        ax.text(n * 1.08, ax.get_ylim()[1] * 0.5 if ax is a2 else 300, "$p=n$", fontsize=7, color=COR["cinza"])
    a1.axhline(beta @ beta + sigma**2, color=CATEG[1], ls="--", lw=1)
    a1.text(2.2, (beta @ beta + sigma**2) * 0.68, "preditor nulo", fontsize=7, color="#333")
    a1.set_ylim(2.2, 3000)
    a1.set_title("(a) risco (mediana e quartis)", fontsize=8.5, loc="left")
    a2.set_title("(b) norma da solução", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "dupla_descida")


# ---------------------------------------------------------------------------
# 4. Gradiente descendente e a solução de norma mínima
# ---------------------------------------------------------------------------
def fig_gradiente():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.array([1.0, 2.0]); yv = 2.0                       # uma equação: b1 + 2 b2 = 2
    t = np.linspace(-1.6, 3.2, 10)
    a1.plot(t, (yv - t) / 2, color=CATEG[0], lw=1.4)
    a1.text(2.15, 0.12, r"$b_1+2b_2=2$", fontsize=7, color="#333")
    bmn = x * yv / (x @ x)
    inits = [np.zeros(2), np.array([-1.2, -0.4]), np.array([1.8, -0.9]), np.array([-0.6, 2.0])]
    for k, b0 in enumerate(inits):
        b = b0.copy(); traj = [b.copy()]
        for _ in range(12):
            b = b - 0.08 * x * (x @ b - yv); traj.append(b.copy())
        traj = np.array(traj)
        cor = CATEG[1] if k == 0 else COR["cinza"]
        a1.plot(*traj.T, "-", color=cor, lw=0.9)
        a1.plot(*b0, "o", ms=3, color=cor)
        lim = b0 - x * (x @ b0 - yv) / (x @ x)
        a1.plot(*lim, "o", ms=3.5, color=cor, markerfacecolor="white")
    a1.plot(*bmn, "o", ms=5, color=CATEG[1])
    a1.annotate(r"$\mathbf{b}^{+}$", bmn, (bmn[0] - 0.9, bmn[1] + 0.7), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=COR["cinza"], lw=0.6))
    a1.set_aspect("equal"); a1.set_xlim(-1.6, 3.2); a1.set_ylim(-1.3, 2.4)
    a1.set_xlabel("$b_1$"); a1.set_ylabel("$b_2$")
    a1.set_title("(a) trajetórias ortogonais à reta", fontsize=8.5, loc="left")

    rng = np.random.default_rng(7)
    n, p = 20, 100
    X = rng.normal(size=(n, p)); y = rng.normal(size=n)
    Xp = np.linalg.pinv(X); b_mn = Xp @ y; eta = 1 / np.linalg.norm(X, 2) ** 2
    b0 = rng.normal(size=p)
    for k, (ini, nome) in enumerate(((np.zeros(p), "início em $\\mathbf{0}$"), (b0, "início aleatório"))):
        b = ini.copy(); d = []
        for _ in range(400):
            d.append(np.linalg.norm(b - b_mn)); b = b - eta * X.T @ (X @ b - y)
        a2.semilogy(d, color=CATEG[[1, 0][k]], lw=1.3)
        a2.text(230, d[-1] * (0.12 if k else 4), nome, fontsize=7, color="#333")
    a2.axhline(np.linalg.norm(b0 - Xp @ X @ b0), color=COR["cinza"], ls=":", lw=0.9)
    a2.set_xlabel("iteração $k$"); a2.set_ylabel(r"$\|\mathbf{b}_k-\mathbf{b}^{+}\|$")
    a2.set_title(r"(b) $n=20$, $p=100$", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "gradiente_norma_minima")


if __name__ == "__main__":
    fig_aveia()
    fig_massey()
    fig_dupla_descida()
    fig_gradiente()
