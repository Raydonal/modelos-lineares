"""Cap. 5: Figuras de dados do capítulo (geradas em figuras/cap05/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import qr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

aplicar_estilo()
CAP = "cap05"
LARG = 5.3
virg = lambda s: s.replace(".", ",")
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "\u2212"))


def virgula(ax, eixos="xy"):
    """Rótulos dos eixos com vírgula decimal."""
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


def proj(X):
    """Projeção ortogonal sobre C(X) (QR com pivotamento; aceita posto incompleto)."""
    Q, R, _ = qr(X, mode="economic", pivoting=True)
    k = (np.abs(np.diag(R)) > 1e-10 * np.abs(R[0, 0])).sum()
    return Q[:, :k] @ Q[:, :k].T


ind = lambda s: pd.get_dummies(s).to_numpy(float)


# ---------------------------------------------------------------------------
# 1. Aveia: somas de quadrados sequenciais, balanceado e com parcelas perdidas
# ---------------------------------------------------------------------------
def fig_aveia():
    aveia = pd.read_csv(DADOS / "yates_oats.csv"); n = len(aveia)
    g = Gerador(5)                                           # mesmos sorteios do script 05
    sorteia = lambda: np.sort(np.argsort(g.unif(n))[:6])
    fatores = {"blocos": "block", "nitrogênio": "nitro", "variedade": "gen"}

    def sq_primeiro_ultimo(d):
        y = d["yield"].to_numpy(float); um = np.ones((len(d), 1))
        X = {f: ind(d[c]) for f, c in fatores.items()}
        res = {}
        for f in fatores:
            outros = np.column_stack([um] + [X[h] for h in fatores if h != f])
            pri = y @ (proj(np.column_stack([um, X[f]])) - proj(um)) @ y
            ult = y @ (proj(np.column_stack([outros, X[f]])) - proj(outros)) @ y
            res[f] = (pri, ult)
        return res

    exemplo = sq_primeiro_ultimo(aveia.drop(index=sorteia()))
    razao = []
    for _ in range(1000):
        r = sq_primeiro_ultimo(aveia.drop(index=sorteia()))["variedade"]
        razao.append(r[1] / r[0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    pos = np.arange(3)
    a1.bar(pos - 0.19, [v[0] for v in exemplo.values()], 0.36, color=CATEG[0], label="fator entra primeiro")
    a1.bar(pos + 0.19, [v[1] for v in exemplo.values()], 0.36, color=CATEG[1], label="fator entra por último")
    a1.set_xticks(pos, list(exemplo.keys()))
    a1.grid(axis="x", visible=False)
    a, b = exemplo["variedade"]
    a1.text(2 - 0.19, a + 400, f"{a:.0f}", ha="center", fontsize=6.5)
    a1.text(2 + 0.19, b + 400, f"{b:.0f}", ha="center", fontsize=6.5)
    a1.set_ylim(0, 24000); a1.set_ylabel("soma de quadrados")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    a1.set_title("(a) um sorteio de 6 parcelas perdidas", fontsize=8.5, loc="left")
    a2.hist(np.log2(razao), bins=40, color=CATEG[0], alpha=0.85)
    a2.axvline(0, color=COR["cinza"], ls=":", lw=0.9)
    a2.set_xticks([-2, -1, 0, 1, 2], ["1/4", "1/2", "1", "2", "4"])
    a2.set_xlabel("SQ(variedade por último) / SQ(variedade primeiro)")
    a2.set_ylabel("sorteios")
    a2.set_title("(b) 1000 sorteios", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "aveia_somas")


# ---------------------------------------------------------------------------
# 2. Variáveis instrumentais: retas de MQ e VI; norma da projeção oblíqua
# ---------------------------------------------------------------------------
def fig_vi():
    m = pd.read_csv(DADOS / "mroz.csv")
    y = m["lwage"].to_numpy(); e = m["educ"].to_numpy(float); n = len(y)
    X = np.column_stack([np.ones(n), e]); Z = np.column_stack([np.ones(n), m["fatheduc"]])
    b_mq = np.linalg.solve(X.T @ X, X.T @ y); b_vi = np.linalg.solve(Z.T @ X, Z.T @ y)
    rng = np.random.default_rng(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.scatter(e + rng.uniform(-0.25, 0.25, n), y, s=4, color=COR["cinza"], alpha=0.45, lw=0)
    g = np.array([e.min(), e.max()])
    a1.plot(g, b_mq[0] + b_mq[1] * g, color=CATEG[0], label=virg(f"MQ: {b_mq[1]:.3f}"))
    a1.plot(g, b_vi[0] + b_vi[1] * g, color=CATEG[1], ls="--", label=virg(f"VI: {b_vi[1]:.3f}"))
    a1.set_xlabel("anos de escolaridade"); a1.set_ylabel("log do salário-hora")
    a1.legend(fontsize=6.5, loc="lower right")
    a1.set_title("(a) retorno da educação", fontsize=8.5, loc="left")

    r = np.linspace(0.08, 1, 200)
    a2.plot(r, 1 / r, color=CATEG[0])
    for nome, cor in (("fatheduc", CATEG[1]), ("motheduc", CATEG[2])):
        rr = np.corrcoef(e, m[nome])[0, 1]
        a2.plot(rr, 1 / rr, "o", ms=4, color=cor)
        a2.annotate(nome, (rr, 1 / rr), xytext=(7, 3) if nome == "fatheduc" else (-38, -11),
                    textcoords="offset points", fontsize=6.5, color=cor)
    a2.set_yscale("log")
    a2.set_xlabel("correlação $r$ entre instrumento e regressor")
    a2.set_ylabel(r"$\|\mathbf{P}\|=1/|r|$")
    a2.set_title("(b) amplificação da projeção oblíqua", fontsize=8.5, loc="left")
    virgula(a2, "x")
    fig.tight_layout()
    salvar(fig, CAP, "vi_mroz")


# ---------------------------------------------------------------------------
# 3. Lovell: série, componente sazonal e coeficientes
# ---------------------------------------------------------------------------
def fig_lovell():
    s = pd.read_csv(DADOS / "seatbelts.csv"); n = len(s)
    y = np.log(s["motoristas"].to_numpy())
    X = np.column_stack([np.log(s["km"]), np.log(s["preco_gasolina"]), s["lei"]])
    D = ind(s["mes"]); PS = D @ np.linalg.solve(D.T @ D, D.T); MS = np.eye(n) - PS
    t = s["ano"] + (s["mes"] - 0.5) / 12
    mq = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
    um = np.ones((n, 1))
    b = {"meses como\nregressores": mq(np.column_stack([X, D]), y)[0],
         "x e y\ndessazonalizados": mq(MS @ X, MS @ y)[0],
         "só y\ndessazonalizado": mq(np.column_stack([um, X]), MS @ y)[1],
         "sazonalidade\nignorada": mq(np.column_stack([um, X]), y)[1]}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.plot(t, y, color=COR["cinza"], lw=0.7, label="log motoristas mortos/feridos")
    a1.plot(t, MS @ y + y.mean(), color=CATEG[0], lw=1.1, label=r"$\mathbf{M}_S\mathbf{y}$ (+ média)")
    a1.axvline(1983 + 1 / 12, color=CATEG[1], ls=":", lw=0.9)
    a1.text(1983.2, y.max() - 0.02, "lei do\ncinto", fontsize=6.5, color=CATEG[1], va="top")
    a1.set_title("(a) série e série dessazonalizada", fontsize=8.5, loc="left")
    a1.legend(fontsize=6.3, loc="lower left")
    pos = np.arange(len(b))
    a2.hlines(pos, 0, list(b.values()), color=COR["cinza"], lw=0.6)
    a2.plot(list(b.values()), pos, "o", ms=4.5, color=CATEG[0])
    a2.plot(list(b.values())[2], 2, "o", ms=4.5, color=CATEG[1])
    a2.set_yticks(pos, list(b.keys()), fontsize=6.3)
    a2.invert_yaxis(); a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.set_xlim(-0.2, 0.01)
    a2.set_xlabel("elasticidade em relação a km")
    a2.set_title("(b) coeficiente de log km", fontsize=8.5, loc="left")
    virgula(a1, "y"); virgula(a2, "x")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    salvar(fig, CAP, "lovell")


# ---------------------------------------------------------------------------
# 4. Suavizadores lineares no mcycle
# ---------------------------------------------------------------------------
def fig_suavizadores():
    mc = pd.read_csv(DADOS / "mcycle.csv"); x = mc["times"].to_numpy(); y = mc["accel"].to_numpy()

    def base(K, xx):
        h = (x.max() - x.min()) / (K + 1)
        return BSpline.design_matrix(np.clip(xx, x.min(), x.max()), x.min() + h * np.arange(-3, K + 5), 3).toarray()

    B9, B20 = base(9, x), base(20, x)
    S_reg = B9 @ np.linalg.solve(B9.T @ B9, B9.T)
    Dd = np.diff(np.eye(B20.shape[1]), n=2, axis=0)
    S_pen = B20 @ np.linalg.solve(B20.T @ B20 + 0.7943282 * Dd.T @ Dd, B20.T)   # lambda do script 08
    W = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.9 ** 2); S_nw = W / W.sum(1, keepdims=True)
    S = {"spline de regressão (gl = 13)": S_reg, "P-spline (gl = 11,3)": S_pen, "Nadaraya–Watson (gl = 23,7)": S_nw}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.scatter(x, y, s=5, color=COR["cinza"], alpha=0.6, lw=0)
    for k, (nome, M) in enumerate(S.items()):
        a1.plot(x, M @ y, color=CATEG[k], lw=1.1, ls=["-", "--", ":"][k], label=nome)
    a1.set_xlabel("tempo após o impacto (ms)"); a1.set_ylabel("aceleração (g)")
    a1.legend(fontsize=6, loc="lower right")
    a1.set_title("(a) três suavizadores", fontsize=8.5, loc="left")
    i = np.argmin(np.abs(x - 30))
    for k, (nome, M) in enumerate(S.items()):
        a2.plot(x, M[i], color=CATEG[k], lw=1.1, ls=["-", "--", ":"][k], marker="o", ms=1.6)
    a2.axhline(0, color=COR["cinza"], lw=0.5)
    a2.axvline(x[i], color=COR["cinza"], ls=":", lw=0.7)
    a2.set_xlabel("tempo (ms)"); a2.set_ylabel(r"pesos $s_{ij}$")
    a2.set_xlim(15, 45)
    a2.set_title(virg(f"(b) linha de S em t = {x[i]:.1f} ms"), fontsize=8.5, loc="left")
    virgula(a2, "y")
    fig.tight_layout()
    salvar(fig, CAP, "suavizadores")


# ---------------------------------------------------------------------------
# 5. Atenção: temperatura e colapso de posto
# ---------------------------------------------------------------------------
def fig_atencao():
    def softmax(M):
        E = np.exp(M - M.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

    g = Gerador(2026)                                        # mesmos sorteios do script 09
    d = 8; grupo = np.repeat([0, 1, 2], [5, 4, 3]); n = len(grupo)
    K = g.matriz_normal(3, d, 1.5)[grupo]
    Q = K + g.matriz_normal(n, d, 0.3)
    A1 = softmax(Q @ K.T / (3 * np.sqrt(d))); A2 = softmax(Q @ K.T / (0.1 * np.sqrt(d)))
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, A, tit in ((axs[0], A1, r"(a) $\tau=3$"), (axs[1], A2, r"(b) $\tau=0{,}1$")):
        ax.imshow(A, cmap="Blues", vmin=0, vmax=0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for c in (4.5, 8.5):
            ax.axhline(c, color="white", lw=0.8); ax.axvline(c, color="white", lw=0.8)
        ax.set_title(tit, fontsize=8.5, loc="left")
    X = g.matriz_normal(n, d)
    curvas = {"tokens genéricos": softmax(X @ X.T / np.sqrt(d)), r"agrupados, $\tau=1$": softmax(Q @ K.T / np.sqrt(d))}
    ks = np.arange(1, 129)
    for c, (nome, A) in enumerate(curvas.items()):
        val, vec = np.linalg.eig(A.T)
        pi = np.real(vec[:, np.argmax(np.real(val))]); pi /= pi.sum(); L = np.outer(np.ones(n), pi)
        dist, Ak = [], np.eye(n)
        for _ in ks:
            Ak = Ak @ A; dist.append(np.abs(Ak - L).max())
        axs[2].semilogy(ks, dist, color=CATEG[c], label=nome)
    axs[2].set_xlabel("número de camadas $k$")
    axs[2].set_ylabel(r"$\max|\mathbf{A}^k-\mathbf{1}\boldsymbol{\pi}^\top|$", fontsize=7)
    axs[2].legend(fontsize=5.8, loc="lower left")
    axs[2].set_title("(c) colapso de posto", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "atencao")


if __name__ == "__main__":
    fig_aveia()
    fig_vi()
    fig_lovell()
    fig_suavizadores()
    fig_atencao()
