"""Cap. 2 -- Figuras de dados do capítulo (geradas em figuras/cap02/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap02"
LARG = 5.3
EPS = np.finfo(float).eps


def matriz_cond(kappa, m=60, n=12, semente=1):
    rng = np.random.default_rng(semente)
    U = np.linalg.qr(rng.normal(size=(m, n)))[0]
    V = np.linalg.qr(rng.normal(size=(n, n)))[0]
    return U @ np.diag(kappa ** (-np.arange(n) / (n - 1))) @ V.T


def gs_classico(X):
    Q = X.copy()
    for j in range(X.shape[1]):
        v = X[:, j] - Q[:, :j] @ (Q[:, :j].T @ X[:, j])
        Q[:, j] = v / np.linalg.norm(v)
    return Q


def gs_modificado(X):
    Q = X.copy()
    for j in range(X.shape[1]):
        Q[:, j] /= np.linalg.norm(Q[:, j])
        Q[:, j + 1:] -= np.outer(Q[:, j], Q[:, j] @ Q[:, j + 1:])
    return Q


# ---------------------------------------------------------------------------
# 1. Estabilidade: ortogonalidade perdida e erro de mínimos quadrados
# ---------------------------------------------------------------------------
def fig_estabilidade():
    kappas = 10.0 ** np.linspace(0, 14, 29)
    perda = lambda Q: np.abs(Q.T @ Q - np.eye(Q.shape[1])).max()
    res = {"cgs": [], "mgs": [], "hh": [], "ne": [], "qr": []}
    rng = np.random.default_rng(3)
    beta = rng.normal(size=12)
    for k in kappas:
        X = matriz_cond(k)
        res["cgs"].append(perda(gs_classico(X)))
        res["mgs"].append(perda(gs_modificado(X)))
        res["hh"].append(perda(np.linalg.qr(X)[0]))
        y = X @ beta
        b_ne = np.linalg.solve(X.T @ X, X.T @ y)
        Q, R = np.linalg.qr(X)
        b_qr = sla.solve_triangular(R, Q.T @ y)
        rel = lambda b: np.linalg.norm(b - beta) / np.linalg.norm(beta)
        res["ne"].append(rel(b_ne)); res["qr"].append(rel(b_qr))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    for chave, rot, cor in (("cgs", "Gram–Schmidt clássico", CATEG[1]),
                            ("mgs", "Gram–Schmidt modificado", CATEG[3]),
                            ("hh", "Householder", CATEG[0])):
        a1.loglog(kappas, np.maximum(res[chave], 1e-17), "o-", ms=2.5, color=cor, label=rot)
    a1.loglog(kappas, np.minimum(EPS * kappas**2, 1), ":", color=COR["cinza"], lw=1)
    a1.loglog(kappas, EPS * kappas, ":", color=COR["cinza"], lw=1)
    a1.text(1.5e4, 1e-5, r"$\kappa^2\epsilon$", fontsize=7, color=COR["cinza"])
    a1.text(3e10, 1e-7, r"$\kappa\epsilon$", fontsize=7, color=COR["cinza"])
    a1.set_xlabel(r"número de condição $\kappa(\mathbf{X})$")
    a1.set_ylabel(r"$\max|\mathbf{Q}^\top\mathbf{Q}-\mathbf{I}|$")
    a1.set_title("(a) perda de ortogonalidade", fontsize=8.5, loc="left")
    a1.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    a2.loglog(kappas, np.minimum(res["ne"], 10), "o-", ms=2.5, color=CATEG[1], label="equações normais")
    a2.loglog(kappas, res["qr"], "o-", ms=2.5, color=CATEG[0], label="QR")
    a2.loglog(kappas, np.minimum(EPS * kappas**2, 10), ":", color=COR["cinza"], lw=1)
    a2.loglog(kappas, EPS * kappas, ":", color=COR["cinza"], lw=1)
    a2.set_xlabel(r"número de condição $\kappa(\mathbf{X})$")
    a2.set_ylabel(r"erro relativo em $\widehat{\boldsymbol{\beta}}$")
    a2.set_title("(b) mínimos quadrados", fontsize=8.5, loc="left")
    a2.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    salvar(fig, CAP, "estabilidade")


# ---------------------------------------------------------------------------
# 2. Longley: algarismos corretos por coeficiente
# ---------------------------------------------------------------------------
def fig_longley():
    lon = pd.read_csv(DADOS / "longley.csv")
    cols = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
    X = np.column_stack([np.ones(len(lon)), lon[cols].to_numpy()])
    y = lon["Employed"].to_numpy()
    cert = np.array([-3482.25863459582, 0.0150618722713733, -0.0358191792925910,
                     -0.0202022980381683, -0.0103322686717359, -0.0511041056535807,
                     1.82915146461355])
    lre = lambda b: np.minimum(-np.log10(np.abs(b - cert) / np.abs(cert)), 15)
    b_ne = np.linalg.solve(X.T @ X, X.T @ y)
    Q, R = np.linalg.qr(X)
    b_qr = sla.solve_triangular(R, Q.T @ y)
    nomes = ["intercepto"] + cols
    fig, ax = plt.subplots(figsize=(LARG, 2.2))
    pos = np.arange(len(nomes))
    ax.bar(pos - 0.2, lre(b_ne), 0.38, color=CATEG[1], label="equações normais")
    ax.bar(pos + 0.2, lre(b_qr), 0.38, color=CATEG[0], label="QR (Householder)")
    ax.axhline(-np.log10(EPS), color=COR["cinza"], ls=":", lw=1)
    ax.text(6.55, -np.log10(EPS) + 0.2, "precisão\nde máquina", fontsize=6.5, color=COR["cinza"], ha="right", va="bottom")
    ax.set_xticks(pos, nomes, fontsize=7, rotation=15)
    ax.set_ylabel("algarismos corretos (LRE)")
    ax.set_ylim(0, 18.5)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    salvar(fig, CAP, "longley_lre")


# ---------------------------------------------------------------------------
# 3. Matriz de delineamento superparametrizada (aveia)
# ---------------------------------------------------------------------------
def fig_delineamento():
    a = pd.read_csv(DADOS / "yates_oats.csv").sort_values(["block", "gen", "nitro"])
    ind = lambda s: pd.get_dummies(s).to_numpy(float)
    X = np.column_stack([np.ones(len(a)), ind(a["block"]), ind(a["nitro"]), ind(a["gen"])])
    fig, ax = plt.subplots(figsize=(LARG * 0.75, 3.4))
    ax.imshow(X, aspect="auto", cmap="Blues", vmin=0, vmax=1.3, interpolation="nearest")
    for x in (0.5, 6.5, 10.5):
        ax.axvline(x, color=COR["destaque"], lw=1.2)
    for xc, rot in ((0, "1"), (3.5, "blocos (6)"), (8.5, "doses N (4)"), (12, "variedades (3)")):
        ax.text(xc, -2.5, rot, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([]); ax.set_ylabel("parcela (72)"); ax.grid(False)
    ax.set_ylim(71.5, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    salvar(fig, CAP, "delineamento_aveia")


# ---------------------------------------------------------------------------
# 4. Posto numérico dos dígitos: diagonal de R no QR com pivotamento
# ---------------------------------------------------------------------------
def fig_digitos_posto():
    dig = pd.read_csv(DADOS / "digitos.csv")
    X = dig.drop(columns="digito").to_numpy(float)
    d = np.abs(np.diag(sla.qr(X, mode="economic", pivoting=True)[1]))
    Xc = X - X.mean(0)
    dc = np.abs(np.diag(sla.qr(Xc, mode="economic", pivoting=True)[1]))
    fig, ax = plt.subplots(figsize=(LARG, 2.3))
    k = np.arange(1, 65)
    ax.semilogy(k, np.maximum(dc, 1e-16), "-", lw=1.2, color=CATEG[2], label="dados centrados")
    ax.semilogy(k, np.maximum(d, 1e-16), "o", ms=2.8, color=CATEG[0], label="dados brutos")
    ax.axhline(1e-2 * d[0], color=COR["cinza"], ls="--", lw=0.9)
    ax.text(30, 1e-2 * d[0] * 0.08, "tolerância relativa $10^{-2}$", fontsize=6.5, color=COR["cinza"], ha="center")
    ax.axvspan(61.5, 64.5, color=CATEG[1], alpha=0.12, lw=0)
    ax.text(60.8, 1e-9, "3 colunas\nnulas", fontsize=7, ha="right", color=CATEG[1])
    ax.set_xlabel("$k$"); ax.set_ylabel(r"$|r_{kk}|$")
    ax.legend(fontsize=7, loc="lower left")
    salvar(fig, CAP, "digitos_posto")


if __name__ == "__main__":
    fig_estabilidade()
    fig_longley()
    fig_delineamento()
    fig_digitos_posto()
