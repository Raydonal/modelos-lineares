"""Cap. 12 -- Figuras de dados do capítulo (geradas em figuras/cap12/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap12"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. Precisão de SF − DAL ao longo da temporada --------------------------------------
def fig_ratings():
    nfl = pd.read_csv(DADOS / "nfl.csv")
    s = nfl[nfl["ano"] == 1981]
    times = sorted(set(s["mandante"]) | set(s["visitante"]))
    idx = {t: i for i, t in enumerate(times)}

    def X_de(j):
        X = np.zeros((len(j), 29)); X[:, 28] = 1
        X[np.arange(len(j)), j["mandante"].map(idx)] = 1
        X[np.arange(len(j)), j["visitante"].map(idx)] = -1
        return X

    l = np.zeros(29); l[idx["SF"]] = 1; l[idx["DAL"]] = -1
    Xf = X_de(s); yf = (s["pontos_mandante"] - s["pontos_visitante"]).to_numpy(float)
    Gf = np.linalg.pinv(Xf.T @ Xf, rcond=1.5e-8)
    r = yf - Xf @ Gf @ Xf.T @ yf
    sd = np.sqrt(r @ r / (len(yf) - 28))
    sem, ep, med = [], [], []
    for w in range(3, 17):
        X = X_de(s[s["semana"] <= w]); G = np.linalg.pinv(X.T @ X, rcond=1.5e-8)
        sem.append(w); ep.append(sd * np.sqrt(l @ G @ l))
        v = [G[i, i] + G[j, j] - 2 * G[i, j] for i in range(28) for j in range(i + 1, 28)]
        med.append(np.mean(sd * np.sqrt(v)))
    sem = np.array(sem)
    fig, ax = plt.subplots(figsize=(0.72 * LARG, 2.5))
    ax.plot(sem, ep, "o-", ms=3.5, color=CATEG[0], label="SF − DAL")
    ax.plot(sem, med, "s-", ms=3, color=CATEG[2], label="média das 378 diferenças")
    ax.plot(sem, sd * np.sqrt(2 / sem), "--", color=COR["cinza"], lw=1,
            label=r"$s\sqrt{2/w}$ (jogos independentes)")
    ax.axvline(2, color=COR["destaque"], lw=0.8, ls=":")
    ax.text(2.2, 20, "semana 2:\nnão estimável", fontsize=7, color=COR["destaque"], va="top")
    ax.set_xlim(1.5, 16.5); ax.set_ylim(0, 21)
    ax.set_xlabel("semana $w$ da temporada"); ax.set_ylabel("erro-padrão (pontos)")
    ax.legend(fontsize=7, frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    virgula(ax)
    salvar(fig, CAP, "ratings_precisao")


# 2. Alocação ao controle num teste A/B/n -------------------------------------------
def fig_alocacao():
    f = np.linspace(0.04, 0.8, 400)
    fig, ax = plt.subplots(figsize=(0.72 * LARG, 2.5))
    for c, k in zip(CATEG, [1, 4, 9]):
        v = 1 / f + k / (1 - f)                # N · Var(ȳ_t − ȳ_0) / σ²
        ax.plot(f, v, color=c, label=f"$k={k}$")
        fo = np.sqrt(k) / (k + np.sqrt(k)); fi = 1 / (k + 1)
        ax.plot(fo, 1 / fo + k / (1 - fo), "o", color=c, ms=4)
        ax.plot(fi, 1 / fi + k / (1 - fi), "o", mfc="white", mec=c, ms=4)
    ax.set_ylim(0, 40)
    ax.set_xlabel(r"fração do total no controle, $n_0/N$")
    ax.set_ylabel(r"$N\,\mathrm{Var}(\bar y_t-\bar y_0)/\sigma^2$")
    ax.legend(fontsize=7, frameon=False, title="variantes", title_fontsize=7)
    ax.text(0.36, 21, "● ótima   ○ igual", fontsize=7, color=COR["cinza"])
    virgula(ax)
    salvar(fig, CAP, "alocacao")


# 3. Retorno da educação por região e combinações ------------------------------------
def fig_federada():
    w = pd.read_csv(DADOS / "wage1.csv")
    w["regiao"] = np.select([w.northcen == 1, w.south == 1, w.west == 1],
                            ["centro-norte", "sul", "oeste"], "nordeste")
    X = np.column_stack([np.ones(len(w)), w.educ, w.exper, w.exper**2 / 100,
                         w.tenure, w.female])
    y = np.log(w.wage.to_numpy()); n, p = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.sum((y - X @ b) ** 2) / (n - p)
    nomes, est, ep = [], [], []
    invs = []
    for k, i in w.groupby("regiao").indices.items():
        A = np.linalg.inv(X[i].T @ X[i]); invs.append(A)
        nomes.append(k); est.append((A @ X[i].T @ y[i])[1]); ep.append(np.sqrt(s2 * A[1, 1]))
    nomes += ["média simples", "federada (= MQ)"]
    est += [np.mean(est), b[1]]
    ep += [np.sqrt(s2 * sum(invs)[1, 1] / 16), np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])]
    est = 100 * np.array(est); ep = 100 * np.array(ep)
    yy = np.arange(len(nomes))[::-1]
    fig, ax = plt.subplots(figsize=(0.72 * LARG, 2.4))
    cores = [COR["cinza"]] * 4 + [CATEG[1], CATEG[0]]
    for yi, e, s_, c in zip(yy, est, ep, cores):
        ax.errorbar(e, yi, xerr=2 * s_, fmt="o", color=c, ms=4, capsize=2, lw=1.2)
    ax.axhline(1.5, color=COR["cinza"], lw=0.5)
    ax.set_yticks(yy); ax.set_yticklabels(nomes)
    ax.set_xlabel("retorno de um ano de estudo (%), ± 2 EP")
    virgula(ax, "x")
    salvar(fig, CAP, "federada")


if __name__ == "__main__":
    fig_ratings()
    fig_alocacao()
    fig_federada()
