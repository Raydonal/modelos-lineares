"""Cap. 9: Figuras de dados do capítulo (geradas em figuras/cap09/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.linalg import cho_solve, cholesky
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

aplicar_estilo()
CAP = "cap09"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. Geometria da normal bivariada ------------------------------------------------
def fig_normal_bivariada():
    S = np.array([[1.0, 0.8], [0.8, 1.0]]) * np.array([[1.0], [1.5]]) * np.array([1.0, 1.5])
    lam, V = np.linalg.eigh(S)
    g = np.linspace(-4.5, 4.5, 300); G1, G2 = np.meshgrid(g, g)
    P = np.stack([G1, G2], -1); Si = np.linalg.inv(S)
    d2 = np.einsum("...i,ij,...j->...", P, Si, P)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6))
    a1.contour(G1, G2, d2, levels=[1, 4, 9], colors=[CATEG[0]], linewidths=1)
    for k in (1, 0):
        v = 2 * V[:, k] * np.sqrt(lam[k]) * np.sign(V[1, k])
        a1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CATEG[3], lw=1.6))
    b = S[1, 0] / S[0, 0]
    a1.plot(g, b * g, color=CATEG[2], lw=1.2, label=r"$\mathrm{E}(x_2\mid x_1)$")
    a1.plot(g, V[1, 1] / V[0, 1] * g, ":", color=CATEG[1], lw=1, label="eixo principal")
    a1.set_xlim(-3.5, 3.5); a1.set_ylim(-4.5, 4.5); a1.set_aspect("equal")
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$x_2$"); a1.legend(fontsize=6.3, loc="lower right")
    a1.set_title("(a) elipses de Mahalanobis 1, 2, 3", fontsize=8.5, loc="left")
    sc = np.sqrt(S[1, 1] - S[1, 0] ** 2 / S[0, 0])
    for k, x1 in enumerate((-1.5, 0.0, 1.5)):
        dens = norm.pdf(g, b * x1, sc)
        a2.plot(g, dens, color=CATEG[k], label=f"$x_1={x1:g}$".replace(".", "{,}"))
    a2.plot(g, norm.pdf(g, 0, np.sqrt(S[1, 1])), "--", color=COR["cinza"], lw=1, label="marginal")
    a2.set_xlabel("$x_2$"); a2.set_ylabel("densidade"); a2.legend(fontsize=6.3, loc="upper left")
    a2.set_ylim(0, 0.6)
    a2.set_title("(b) condicionais de $x_2$", fontsize=8.5, loc="left"); virgula(a2)
    virgula(a1)
    fig.tight_layout(); salvar(fig, CAP, "normal_bivariada")


# 2. Carteira de variância mínima -------------------------------------------------
def fig_carteira():
    eu = pd.read_csv(DADOS / "eustock.csv"); nomes = list(eu.columns[1:])
    r = 100 * np.diff(np.log(eu.iloc[:, 1:].to_numpy()), axis=0)
    m = r.mean(0); S = np.cov(r.T); um = np.ones(4)
    a = um @ np.linalg.solve(S, um); b = um @ np.linalg.solve(S, m); c = m @ np.linalg.solve(S, m)
    t = np.linspace(0.02, 0.12, 200); dp = np.sqrt((a * t ** 2 - 2 * b * t + c) / (a * c - b ** 2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6))
    a1.plot(dp, t, color=CATEG[0], label="fronteira")
    a1.plot(np.sqrt(np.diag(S)), m, "o", ms=4, color=CATEG[1])
    for k, nm in enumerate(nomes):
        a1.annotate(nm, (np.sqrt(S[k, k]), m[k]), xytext=(4, -2), textcoords="offset points", fontsize=6.5)
    a1.plot(np.sqrt(1 / a), b / a, "s", ms=4, color=CATEG[2], label="variância mínima")
    a1.plot(np.sqrt(um @ S @ um / 16), m.mean(), "^", ms=4, color=CATEG[3], label="pesos iguais")
    a1.set_xlabel("desvio-padrão diário (%)"); a1.set_ylabel("retorno médio diário (%)")
    a1.legend(fontsize=6.3, loc="lower right"); virgula(a1)
    a1.set_title("(a) fronteira de Markowitz", fontsize=8.5, loc="left")
    w = np.linalg.solve(S, um); w /= w.sum(); rp = r @ w
    h, bordas = np.histogram(rp, bins=60, density=True); cen = (bordas[1:] + bordas[:-1]) / 2
    a2.semilogy(cen[h > 0], h[h > 0], "o", ms=2.2, color=CATEG[0], label="empírica")
    gg = np.linspace(rp.min(), rp.max(), 300)
    a2.semilogy(gg, norm.pdf(gg, rp.mean(), rp.std(ddof=1)), color=CATEG[1], label="normal")
    a2.set_ylim(1e-3, 1.5); a2.set_xlabel("retorno da carteira (%)"); a2.set_ylabel("densidade (escala log)")
    a2.legend(fontsize=6.3); virgula(a2, "x")
    a2.set_title("(b) caudas", fontsize=8.5, loc="left")
    fig.tight_layout(); salvar(fig, CAP, "carteira")


# 3. Imputação condicional --------------------------------------------------------
def fig_imputacao():
    aq = pd.read_csv(DADOS / "airquality.csv")
    V = np.column_stack([np.log(aq["Ozone"]), aq["Solar.R"], aq["Wind"], aq["Temp"]])
    cc = ~np.isnan(V).any(1); mu = V[cc].mean(0); S = np.cov(V[cc].T)
    fig, ax = plt.subplots(figsize=(0.7 * LARG, 2.6))
    ax.plot(V[cc, 3], V[cc, 0], "o", ms=2.5, color=CATEG[0], alpha=0.7, label="observado")
    for i in np.where(np.isnan(V[:, 0]))[0]:
        m = np.isnan(V[i]); o = ~m
        B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
        med = mu[m] + B @ (V[i, o] - mu[o]); cv = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
        ax.errorbar(V[i, 3], med[0], yerr=1.645 * np.sqrt(cv[0, 0]), fmt="s", ms=2.8,
                    color=CATEG[1], elinewidth=0.7, capsize=0)
    ax.errorbar([], [], yerr=[], fmt="s", ms=2.8, color=CATEG[1], label="imputado (90%)")
    ax.set_xlabel("temperatura (°F)"); ax.set_ylabel("log(ozônio)"); ax.legend(fontsize=6.3, loc="upper left")
    virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "imputacao")


# 4. Anomalias nos dígitos --------------------------------------------------------
def fig_anomalias():
    dig = pd.read_csv(DADOS / "digitos.csv")
    X = dig.iloc[:, 1:].to_numpy() / 16; y = dig["digito"].to_numpy()
    tr, te = np.arange(1000), np.arange(1000, len(y))
    Z = X[tr][y[tr] == 1]; mu = Z.mean(0); S = np.cov(Z.T)
    D = X[te] - mu; anom = y[te] != 1

    def roc(s):
        o = np.argsort(-s); a = anom[o]
        return np.r_[0, np.cumsum(~a) / (~a).sum()], np.r_[0, np.cumsum(a) / a.sum()]
    fig = plt.figure(figsize=(LARG, 2.5))
    a1 = fig.add_axes([0.08, 0.17, 0.38, 0.72])
    for k, (nome, Si) in enumerate((("euclidiana", np.eye(64)),
                                    ("Mahalanobis", np.linalg.inv(S + 0.01 * np.eye(64))))):
        f, v = roc(np.sum((D @ Si) * D, 1)); a1.plot(f, v, color=CATEG[k], label=nome)
    a1.set_xlabel("falsos positivos"); a1.set_ylabel("verdadeiros positivos")
    a1.legend(fontsize=6.3, loc="lower right"); virgula(a1)
    a1.set_title("(a) curvas ROC", fontsize=8.5, loc="left")
    lam, Vv = np.linalg.eigh(S)
    imgs = [(mu, "média"), (Vv[:, -1], "1ª direção"), (Vv[:, -2], "2ª direção")]
    for k, (im, tit) in enumerate(imgs):
        ax = fig.add_axes([0.53 + k * 0.155, 0.3, 0.14, 0.5])
        lim = np.abs(im).max()
        ax.imshow(im.reshape(8, 8), cmap="gray_r" if k == 0 else "RdBu_r",
                  vmin=0 if k == 0 else -lim, vmax=lim)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(tit, fontsize=7.5)
    fig.text(0.53, 0.89, "(b) dígitos 1: média e direções de maior variância", fontsize=8.5)
    salvar(fig, CAP, "anomalias")


# 5. Processo gaussiano -----------------------------------------------------------
def fig_processo_gaussiano():
    mc = pd.read_csv(DADOS / "mcycle.csv"); t = mc["times"].to_numpy(); y = mc["accel"].to_numpy()
    n = len(y); r = y - y.mean(); l, tau = 5.25, 10 ** -0.6
    nuc = lambda a, b: np.exp(-np.subtract.outer(a, b) ** 2 / (2 * l ** 2))
    Ac = cholesky(nuc(t, t) + tau * np.eye(n)); s2 = r @ cho_solve((Ac, False), r) / n
    ts = np.linspace(0, 60, 400); Cs = nuc(ts, t)
    med = y.mean() + Cs @ cho_solve((Ac, False), r)
    vf = s2 * (1 - np.sum(Cs * cho_solve((Ac, False), Cs.T).T, 1))
    fig, ax = plt.subplots(figsize=(0.75 * LARG, 2.6))
    ax.fill_between(ts, med - 1.96 * np.sqrt(vf + s2 * tau), med + 1.96 * np.sqrt(vf + s2 * tau),
                    color=CATEG[0], alpha=0.12, lw=0, label="predição de nova medida (95%)")
    ax.fill_between(ts, med - 1.96 * np.sqrt(vf), med + 1.96 * np.sqrt(vf), color=CATEG[0], alpha=0.3,
                    lw=0, label="banda para $f$ (95%)")
    ax.plot(ts, med, color=CATEG[0], label="média a posteriori")
    ax.plot(t, y, "o", ms=2.2, color=CATEG[1])
    ax.set_xlabel("tempo após o impacto (ms)"); ax.set_ylabel("aceleração (g)")
    ax.legend(fontsize=6.3, loc="lower right"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "processo_gaussiano")


# 6. Inicialização ----------------------------------------------------------------
def fig_inicializacao():
    g = Gerador(17); n, L, B = 128, 30, 16
    X0 = g.matriz_normal(n, B); G = [g.matriz_normal(n, n) for _ in range(L)]
    relu = lambda z: np.maximum(z, 0)
    redes = [("linear, $s^2=1$", lambda z: z, 1), ("ReLU, $s^2=1$", relu, 1),
             ("ReLU, $s^2=2$ (He)", relu, 2), ("tanh, $s^2=1$", np.tanh, 1)]
    fig, ax = plt.subplots(figsize=(0.75 * LARG, 2.6))
    for k, (nome, f, s2) in enumerate(redes):
        X = X0; q = [np.mean(X ** 2)]
        for l in range(L):
            X = f(np.sqrt(s2 / n) * G[l] @ X); q.append(np.mean(X ** 2))
        ax.semilogy(range(L + 1), q, "o-", ms=2, color=CATEG[k], label=nome)
    ax.set_xlabel("camada $l$"); ax.set_ylabel("média de $x^2$ na camada")
    ax.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout(); salvar(fig, CAP, "inicializacao")


# 7. PCA probabilística / VAE linear --------------------------------------------
def fig_autocodificador():
    dig = pd.read_csv(DADOS / "digitos.csv"); X = dig.iloc[:, 1:].to_numpy() / 16
    tr, te = X[:1000], X[1000:]; n, p = tr.shape; mu = tr.mean(0)
    S = (tr - mu).T @ (tr - mu) / n; d, Q = np.linalg.eigh(S); d, Q = np.maximum(d[::-1], 0), Q[:, ::-1]

    def ll(k, R):
        s2 = d[k:].mean(); W = Q[:, :k] * np.sqrt(d[:k] - s2)
        G = np.eye(k) + W.T @ W / s2; U = R @ W
        quad = (np.sum(R ** 2, 1) - np.sum(U @ np.linalg.inv(G) * U, 1) / s2) / s2
        return np.mean(-0.5 * (p * np.log(2 * np.pi) + p * np.log(s2) + np.linalg.slogdet(G)[1] + quad))
    ks = np.arange(1, 56)
    fig = plt.figure(figsize=(LARG, 2.6))
    a1 = fig.add_axes([0.08, 0.17, 0.38, 0.72])
    a1.plot(ks, [ll(k, tr - mu) for k in ks], color=CATEG[0], label="treino")
    a1.plot(ks, [ll(k, te - mu) for k in ks], color=CATEG[1], label="teste")
    a1.set_ylim(-20, 80); a1.set_xlabel("dimensão latente $k$"); a1.set_ylabel("log-verossimilhança média")
    a1.legend(fontsize=6.3, loc="upper left"); a1.set_title("(a) PCA probabilística", fontsize=8.5, loc="left")
    g = Gerador(23)
    for lin, k in enumerate((10, 40)):
        s2 = d[k:].mean(); W = Q[:, :k] * np.sqrt(d[:k] - s2)
        for c in range(6):
            ax = fig.add_axes([0.52 + c * 0.077, 0.47 - lin * 0.33, 0.07, 0.3])
            img = mu + W @ g.normal(k)
            ax.imshow(np.clip(img, 0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c == 0:
                ax.set_ylabel(f"$k={k}$", fontsize=7.5)
    fig.text(0.52, 0.89, "(b) dígitos gerados: $\\mu+\\mathbf{W}\\mathbf{z}$, $\\mathbf{z}\\sim N(0,I)$", fontsize=8.5)
    salvar(fig, CAP, "autocodificador")


if __name__ == "__main__":
    fig_normal_bivariada()
    fig_carteira()
    fig_imputacao()
    fig_anomalias()
    fig_processo_gaussiano()
    fig_inicializacao()
    fig_autocodificador()
