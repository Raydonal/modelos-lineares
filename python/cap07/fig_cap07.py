"""Cap. 7: Figuras de dados do capítulo (geradas em figuras/cap07/)."""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

aplicar_estilo()
CAP = "cap07"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# ---------------------------------------------------------------------------
# 1. Longley: efeito de perturbações do tamanho do arredondamento
# ---------------------------------------------------------------------------
def fig_longley():
    L = pd.read_csv(DADOS / "longley.csv")
    nomes = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
    rot = ["intercepto", "deflator", "PIB", "desemprego", "forças arm.", "população", "ano"]
    X = np.column_stack([np.ones(16), L[nomes]]); y = L["Employed"].to_numpy()
    mq = lambda A: np.linalg.lstsq(A, y, rcond=None)[0]
    b = mq(X); casas = np.array([1, 3, 1, 1, 3, 0]); g = Gerador(76); R = []
    for _ in range(1000):
        Xp = X.copy(); Xp[:, 1:] += (g.unif(96) - 0.5).reshape(6, 16).T * 10.0 ** -casas; R.append(mq(Xp))
    R = np.array(R)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5), gridspec_kw={"width_ratios": [1, 1.2]})
    a1.hist(np.linalg.norm(R - b, axis=1) / np.linalg.norm(b), bins=35, color=CATEG[0])
    a1.set_xlabel(r"$\|\tilde{\mathbf{b}}-\widehat{\mathbf{b}}\|/\|\widehat{\mathbf{b}}\|$"); a1.set_ylabel("sorteios")
    a1.set_title("(a) variação relativa dos coeficientes", fontsize=8.5, loc="left"); virgula(a1, "x")
    fl = (np.sign(R) != np.sign(b)).mean(0)
    a2.barh(np.arange(7), fl, color=[CATEG[1] if f > 0.5 else CATEG[0] for f in fl])
    a2.set_yticks(np.arange(7), rot, fontsize=7); a2.invert_yaxis(); a2.set_xlim(0, 1)
    a2.set_xlabel("fração dos sorteios com mudança de sinal"); a2.grid(axis="y", visible=False)
    a2.set_title("(b) sinais instáveis", fontsize=8.5, loc="left"); virgula(a2, "x")
    fig.tight_layout(); salvar(fig, CAP, "longley_arredondamento")


# ---------------------------------------------------------------------------
# 2. Instrumentos fracos
# ---------------------------------------------------------------------------
def fig_instrumentos():
    g = Gerador(77); n, beta = 428, 0.06

    def simula(pi1):
        z, u, v = g.normal(n), g.normal(n), g.normal(n)
        x = pi1 * z + 0.6 * u + 0.8 * v; y = beta * x + 0.5 * u + 0.3 * g.normal(n)
        zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
        return zc @ yc / (zc @ xc)
    forcas = (0.4, 0.2, 0.1, 0.05)
    est = [np.array([simula(p) for _ in range(2000)]) for p in forcas]
    fig, ax = plt.subplots(figsize=(LARG, 2.4))
    bp = ax.boxplot(est, orientation="horizontal", whis=(5, 95), showfliers=False, patch_artist=True, widths=0.5)
    for k, box in enumerate(bp["boxes"]):
        box.set_facecolor(CATEG[k]); box.set_alpha(0.55)
    ax.axvline(beta, color=COR["cinza"], ls=":", lw=0.9)
    ax.set_yticks(range(1, 5), [f"$r\\approx{p:.2f}$".replace(".", "{,}") for p in (0.37, 0.20, 0.10, 0.05)])
    ax.set_xlim(-2.6, 2.6); ax.set_xlabel("estimativa de VI de $\\beta$ (verdadeiro: $0{,}06$)")
    ax.grid(axis="y", visible=False); virgula(ax, "x")
    fig.tight_layout(); salvar(fig, CAP, "instrumentos_fracos")


# ---------------------------------------------------------------------------
# 3. Filtros espectrais nos dígitos
# ---------------------------------------------------------------------------
def fig_filtros():
    dig = pd.read_csv(DADOS / "digitos.csv")
    X = np.column_stack([np.ones(len(dig)), dig.drop(columns="digito").to_numpy() / 16]); y = dig["digito"].to_numpy()
    tr, te = np.arange(1000), np.arange(1000, len(y)); Y = (y[tr, None] == np.arange(10)).astype(float)
    U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
    acc = lambda f: np.mean((X[te] @ (Vt.T @ (f[:, None] * (U.T @ Y)))).argmax(1) == y[te])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    sg = s[:62]
    a1.semilogx(sg, np.where(np.arange(62) < 30, 1.0, 0.0), "o", ms=2.5, color=CATEG[0], label="SVD truncada ($k=30$)")
    a1.semilogx(sg, sg ** 2 / (sg ** 2 + 10), "s", ms=2.5, color=CATEG[1], label="ridge ($\\lambda=10$)")
    a1.set_ylim(-0.05, 1.1); a1.set_xlabel("$\\sigma_j$"); a1.set_ylabel("fator de filtro $\\sigma_jf_j$")
    a1.legend(fontsize=6.3, loc="center left"); a1.set_title("(a) filtros", fontsize=8.5, loc="left"); virgula(a1, "y")
    ks = np.arange(2, 63); gl_r = []; acc_r = []
    for lam in np.logspace(-3, 3, 40):
        gl_r.append((s ** 2 / (s ** 2 + lam)).sum()); acc_r.append(acc(s / (s ** 2 + lam)))
    a2.plot(ks, [acc(np.r_[1 / s[:k], np.zeros(65 - k)]) for k in ks], color=CATEG[0], label="SVD truncada")
    a2.plot(gl_r, acc_r, color=CATEG[1], label="ridge")
    a2.set_ylim(0.6, 0.92); a2.set_xlabel("graus de liberdade efetivos"); a2.set_ylabel("acurácia no teste")
    a2.legend(fontsize=6.3, loc="lower right"); a2.set_title("(b) acurácia", fontsize=8.5, loc="left"); virgula(a2, "y")
    fig.tight_layout(); salvar(fig, CAP, "filtros_espectrais")


# ---------------------------------------------------------------------------
# 4. Quantização dos embeddings
# ---------------------------------------------------------------------------
def fig_quantizacao():
    d = pd.read_csv(DADOS / "austen.csv", keep_default_na=False)
    p = re.findall(r"[a-z]+", " ".join(d["texto"]).lower())
    fr = sorted(Counter(p).items(), key=lambda kv: (-kv[1], kv[0])); V = 2000
    pos = {w: i for i, (w, _) in enumerate(fr[:V])}
    ids = np.array([pos.get(w, -1) for w in p]); C = np.zeros((V, V))
    for o in (1, 2):
        i, j = ids[:-o], ids[o:]; ok = (i >= 0) & (j >= 0); np.add.at(C, (i[ok], j[ok]), 1)
    C = C + C.T; N = C.sum(); pc = C.sum(0) ** 0.75; pc /= pc.sum()
    with np.errstate(divide="ignore"):
        M = np.log((C / N) / np.outer(C.sum(1) / N, pc))
    M[~np.isfinite(M) | (M < 0)] = 0
    U, s, _ = np.linalg.svd(M); W = U[:, :100] * np.sqrt(s[:100])

    def quant(bits, pl):
        q = 2 ** (bits - 1) - 1
        esc = (np.abs(W).max(1, keepdims=True) if pl else np.abs(W).max()) / q
        return np.round(W / esc) * esc

    def viz(A):
        nr = np.linalg.norm(A, axis=1, keepdims=True); nr[nr == 0] = 1
        S = (A / nr)[:300] @ (A / nr).T; S[np.arange(300), np.arange(300)] = -np.inf
        return np.argsort(-S, axis=1, kind="stable")[:, :5]
    v0 = viz(W); bits = np.arange(2, 9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.4))
    for c, (pl, nome) in enumerate(((True, "escala por linha"), (False, "escala global"))):
        e, ov = [], []
        for b in bits:
            Q = quant(b, pl); e.append(np.linalg.norm(W - Q, 2) / np.linalg.norm(W, 2))
            ov.append(np.mean([len(set(x) & set(z)) / 5 for x, z in zip(v0, viz(Q))]))
        a1.semilogy(bits, e, "o-", ms=3, color=CATEG[c], label=nome)
        a2.plot(bits, ov, "o-", ms=3, color=CATEG[c], label=nome)
    a1.set_xlabel("bits"); a1.set_ylabel("$\\|\\mathbf{W}-Q(\\mathbf{W})\\|_2/\\|\\mathbf{W}\\|_2$"); a1.legend(fontsize=6.3)
    a1.set_title("(a) erro espectral relativo", fontsize=8.5, loc="left")
    a2.set_xlabel("bits"); a2.set_ylabel("vizinhos preservados"); a2.set_ylim(0, 1.02); virgula(a2, "y")
    a2.set_title("(b) 5 vizinhos mais próximos", fontsize=8.5, loc="left")
    fig.tight_layout(); salvar(fig, CAP, "quantizacao")


# ---------------------------------------------------------------------------
# 5. Robustez adversarial do classificador linear de dígitos
# ---------------------------------------------------------------------------
def fig_robustez():
    dig = pd.read_csv(DADOS / "digitos.csv")
    X = np.column_stack([np.ones(len(dig)), dig.drop(columns="digito").to_numpy() / 16]); y = dig["digito"].to_numpy()
    tr, te = np.arange(1000), np.arange(1000, len(y)); Y = (y[tr, None] == np.arange(10)).astype(float)
    U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
    filtro = lambda f: Vt.T @ (f[:, None] * (U.T @ Y))
    modelos = {"pseudo-inversa": filtro(np.r_[1 / s[:62], np.zeros(3)]),
               "SVD truncada ($k=30$)": filtro(np.r_[1 / s[:30], np.zeros(35)]),
               "ridge ($\\lambda=10$)": filtro(s / (s ** 2 + 10))}
    n = len(te); ii = np.arange(n); eps = np.linspace(0, 0.25, 51)
    fig, ax = plt.subplots(figsize=(0.75 * LARG, 2.6))
    for k, (nome, W) in enumerate(modelos.items()):
        S = X[te] @ W; c = S.argmax(1); P = W[1:]
        D = P[:, c].T[:, :, None] - P[None]; M = S[ii, c][:, None] - S
        l1 = np.abs(D).sum(1); l1[ii, c] = 1; M[ii, c] = np.inf
        R1 = M / l1; r = R1.min(1); j = R1.argmin(1); ok = c == y[te]
        cert = [np.mean(ok & (r > e)) for e in eps]
        atq = []
        for e in eps:
            Xa = X[te].copy(); Xa[:, 1:] = np.clip(Xa[:, 1:] - e * np.sign(D[ii, :, j]), 0, 1)
            atq.append(np.mean((Xa @ W).argmax(1) == y[te]))
        ax.plot(eps, cert, color=CATEG[k], label=nome)
        ax.plot(eps, atq, "--", color=CATEG[k], lw=1)
    ax.plot([], [], "k-", lw=1, label="certificada"); ax.plot([], [], "k--", lw=1, label="sob ataque")
    ax.set_xlabel("$\\varepsilon$ (perturbação máxima por pixel, escala $[0,1]$)"); ax.set_ylabel("acurácia no teste")
    ax.set_ylim(0, 0.95); ax.legend(fontsize=6.3, loc="upper right"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "robustez")


if __name__ == "__main__":
    fig_longley()
    fig_instrumentos()
    fig_filtros()
    fig_quantizacao()
    fig_robustez()
