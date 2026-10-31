"""Cap. 1: Figuras de dados do capítulo (geradas em figuras/cap01/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap01"
LARG = 5.3                                  # largura do texto (polegadas)


def seta(ax, ini, fim, cor, **kw):
    ax.add_patch(FancyArrowPatch(ini, fim, arrowstyle="-|>", mutation_scale=10,
                                 color=cor, lw=1.6, shrinkA=0, shrinkB=0, **kw))


# ---------------------------------------------------------------------------
# 1. Correlação = cosseno do ângulo entre vetores centrados
# ---------------------------------------------------------------------------
def fig_correlacao_cosseno():
    rng = np.random.default_rng(11)
    alvos = [0.9, 0.5, 0.0, -0.6]
    fig, eixos = plt.subplots(2, 4, figsize=(LARG, 2.9),
                              gridspec_kw={"height_ratios": [1.15, 1]})
    for j, rho in enumerate(alvos):
        z = rng.standard_normal((80, 2))
        x = z[:, 0]
        y = rho * x + np.sqrt(1 - rho**2) * z[:, 1]
        r = np.corrcoef(x, y)[0, 1]
        ax = eixos[0, j]
        ax.scatter(x, y, s=7, color=CATEG[0], alpha=0.75, lw=0)
        ax.set_title(f"$r = {r:.2f}$", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("$x$", fontsize=8, labelpad=1); ax.set_ylabel("$y$", fontsize=8, labelpad=1)
        # diagrama do ângulo entre x - x̄ e y - ȳ
        ax = eixos[1, j]
        th = np.arccos(np.clip(r, -1, 1))
        seta(ax, (0, 0), (1, 0), CATEG[0])
        seta(ax, (0, 0), (np.cos(th), np.sin(th)), CATEG[1])
        ax.add_patch(Wedge((0, 0), 0.32, 0, np.degrees(th), color=COR["ouro"], alpha=0.35, lw=0))
        ax.text(0.5, -0.16, r"$\mathbf{x}-\bar{x}\mathbf{1}$", fontsize=7, ha="center", va="top", color="#333")
        ax.text(np.cos(th) * 1.08, np.sin(th) * 1.08 + 0.04, r"$\mathbf{y}-\bar{y}\mathbf{1}$",
                fontsize=7, ha="center", color="#333")
        ax.text(0.42 * np.cos(th / 2), 0.42 * np.sin(th / 2), f"{np.degrees(th):.0f}°",
                fontsize=7, ha="left", va="bottom", color="#333")
        ax.set_xlim(-1.15, 1.25); ax.set_ylim(-0.35, 1.25)
        ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(h_pad=0.4)
    salvar(fig, CAP, "correlacao_cosseno")


# ---------------------------------------------------------------------------
# 2. Decomposição ortogonal: experimento de aveia de Yates
# ---------------------------------------------------------------------------
def fig_aveia():
    d = pd.read_csv(DADOS / "yates_oats.csv")
    y = d["yield"].to_numpy(float)
    doses = np.sort(d["nitro"].unique())
    medias = d.groupby("nitro")["yield"].mean().reindex(doses).to_numpy()
    yb = y.mean()
    y_trt = d["nitro"].map(dict(zip(doses, medias))).to_numpy()
    sq_tot, sq_trt, sq_res = ((y - yb) ** 2).sum(), ((y_trt - yb) ** 2).sum(), ((y - y_trt) ** 2).sum()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(3)
    for k, (dose, m) in enumerate(zip(doses, medias)):
        yk = y[d["nitro"].to_numpy() == dose]
        a1.scatter(np.full(len(yk), k) + rng.uniform(-0.15, 0.15, len(yk)), yk,
                   s=9, color=CATEG[0], alpha=0.6, lw=0)
        a1.plot([k - 0.3, k + 0.3], [m, m], color=CATEG[1], lw=2.2)
    a1.axhline(yb, color=COR["cinza"], lw=1, ls="--")
    a1.text(3.45, yb, r"$\bar{y}$", va="center", fontsize=8, color=COR["cinza"])
    a1.text(3.45, medias[-1], r"$\bar{y}_{4}$", va="center", fontsize=8, color=CATEG[1])
    a1.set_xticks(range(4), [f"{v:g}" for v in doses])
    a1.set_xlabel("dose de nitrogênio (cwt/acre)")
    a1.set_ylabel("produção (1/4 lb por parcela)")
    a1.set_xlim(-0.5, 3.8)
    a1.set_title("(a) dados, médias por dose e média geral", fontsize=8.5, loc="left")

    # triângulo retângulo em escala: catetos sqrt(SQ_trat), sqrt(SQ_res)
    c1, c2 = np.sqrt(sq_trt), np.sqrt(sq_res)
    a2.add_patch(Polygon([[0, 0], [c2, 0], [c2, c1]], closed=True,
                         fc=COR["fundo"], ec=COR["cinza"], lw=0.8))
    a2.plot([0, c2], [0, 0], color=CATEG[2], lw=2.6)
    a2.plot([c2, c2], [0, c1], color=CATEG[1], lw=2.6)
    a2.plot([0, c2], [0, c1], color=CATEG[0], lw=2.6)
    s = 9
    a2.plot([c2 - s, c2 - s, c2], [0, s, s], color=COR["cinza"], lw=0.8)
    a2.text(c2 / 2, -14, f"resíduo\n$\\sqrt{{{sq_res:,.0f}}}$".replace(",", "."), ha="center", va="top", fontsize=7.5)
    a2.text(c2 + 6, c1 / 2, f"tratamento\n$\\sqrt{{{sq_trt:,.0f}}}$".replace(",", "."), ha="left", va="center", fontsize=7.5)
    a2.text(c2 / 2 - 14, c1 / 2 + 8, f"total\n$\\sqrt{{{sq_tot:,.0f}}}$".replace(",", "."), ha="right", va="bottom", fontsize=7.5)
    a2.set_xlim(-60, c2 + 90); a2.set_ylim(-60, c1 + 25)
    a2.set_aspect("equal"); a2.axis("off")
    a2.set_title("(b) Pitágoras: SQ$_{tot}$ = SQ$_{trat}$ + SQ$_{res}$", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "aveia_decomposicao")


# ---------------------------------------------------------------------------
# 3. Dígitos: centróides e matriz de cossenos
# ---------------------------------------------------------------------------
def fig_digitos():
    d = pd.read_csv(DADOS / "digitos.csv")
    X = d.drop(columns="digito").to_numpy(float)[:1200]
    rot = d["digito"].to_numpy()[:1200]
    C = np.vstack([X[rot == k].mean(axis=0) for k in range(10)])
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    S = Cn @ Cn.T

    fig = plt.figure(figsize=(LARG, 2.55))
    gs = fig.add_gridspec(2, 9, width_ratios=[1] * 5 + [0.35, 2.6, 0.12, 0.12], hspace=0.05, wspace=0.08)
    for k in range(10):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.imshow(C[k].reshape(8, 8), cmap="Blues", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(str(k), fontsize=8, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = fig.add_subplot(gs[:, 6])
    im = ax.imshow(S, cmap="Blues", vmin=0.5, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.tick_params(labelsize=7, length=0)
    ax.grid(False)
    off = S - np.eye(10) * 2
    i, j = np.unravel_index(off.argmax(), S.shape)
    for a, b in ((i, j), (j, i)):
        ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=COR["destaque"], lw=1.5))
    ax.set_title("cosseno entre centróides", fontsize=8.5)
    cax = fig.add_subplot(gs[:, 8])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    salvar(fig, CAP, "digitos_cosseno")


# ---------------------------------------------------------------------------
# 4. Camadas: linear, composição linear e ReLU
# ---------------------------------------------------------------------------
def fig_camadas():
    W1 = np.array([[1.2, 0.6], [-0.3, 0.9]])
    W2 = np.array([[0.8, -0.7], [0.5, 0.6]])
    t = np.linspace(-1, 1, 9)
    s = np.linspace(-1, 1, 200)
    linhas = [np.column_stack([np.full_like(s, a), s]) for a in t] + \
             [np.column_stack([s, np.full_like(s, a)]) for a in t]
    circ = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 300)), np.sin(np.linspace(0, 2 * np.pi, 300))])
    mapas = [
        (r"entrada $\mathbf{x}$", lambda P: P),
        (r"$\mathbf{W}_1\mathbf{x}$", lambda P: P @ W1.T),
        (r"$\mathbf{W}_2\mathbf{W}_1\mathbf{x}$", lambda P: P @ (W2 @ W1).T),
        (r"$\mathrm{ReLU}(\mathbf{W}_1\mathbf{x})$", lambda P: np.maximum(P @ W1.T, 0)),
    ]
    fig, eixos = plt.subplots(1, 4, figsize=(LARG, 1.6))
    for ax, (tit, f) in zip(eixos, mapas):
        for L in linhas:
            Q = f(L)
            ax.plot(Q[:, 0], Q[:, 1], color=CATEG[0], lw=0.5, alpha=0.6)
        Q = f(circ)
        ax.plot(Q[:, 0], Q[:, 1], color=CATEG[1], lw=1.6)
        for e, cor in ((np.array([1.0, 0]), CATEG[2]), (np.array([0, 1.0]), CATEG[3])):
            fe = f(e[None, :])[0]
            seta(ax, (0, 0), tuple(fe), cor)
        ax.set_title(tit, fontsize=8.5)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color="#BBB", lw=0.5); ax.axvline(0, color="#BBB", lw=0.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    salvar(fig, CAP, "camadas_lineares")


# ---------------------------------------------------------------------------
# 5. Bolas unitárias das normas l_p
# ---------------------------------------------------------------------------
def fig_bolas():
    ps = [(0.5, r"$p=1/2$ (não é norma)"), (1, r"$p=1$"), (2, r"$p=2$"), (4, r"$p=4$"), (np.inf, r"$p=\infty$")]
    th = np.linspace(0, 2 * np.pi, 800)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    cores = [COR["cinza"]] + CATEG[:4]
    for (p, rot), cor in zip(ps, cores):
        u = np.column_stack([np.cos(th), np.sin(th)])
        if np.isinf(p):
            nrm = np.abs(u).max(axis=1)
        else:
            nrm = (np.abs(u) ** p).sum(axis=1) ** (1 / p)
        B = u / nrm[:, None]
        ax.plot(B[:, 0], B[:, 1], color=cor, lw=1.8 if p >= 1 else 1.2, ls="-" if p >= 1 else "--", label=rot)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.axhline(0, color="#BBB", lw=0.5); ax.axvline(0, color="#BBB", lw=0.5)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    salvar(fig, CAP, "bolas_lp")


# ---------------------------------------------------------------------------
# 6. Retorno da educação: reta ajustada e decomposição de Pitágoras
# ---------------------------------------------------------------------------
def fig_salario():
    d = pd.read_csv(DADOS / "wage1.csv")
    y = np.log(d["wage"].to_numpy()); x = d["educ"].to_numpy(float)
    b1 = ((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean()))
    b0 = y.mean() - b1 * x.mean()
    rng = np.random.default_rng(5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.scatter(x + rng.uniform(-0.25, 0.25, len(x)), y, s=5, alpha=0.4, color=CATEG[0], lw=0)
    xs = np.linspace(0, 18, 50)
    a1.plot(xs, b0 + b1 * xs, color=CATEG[1], lw=2)
    a1.text(1, 2.9, f"$\\hat y = {b0:.3f} + {b1:.4f}\\,x$", fontsize=8, color="#333")
    a1.set_xlabel("anos de escolaridade"); a1.set_ylabel("log(salário-hora)")
    a1.set_title("(a) dados e reta de mínimos quadrados", fontsize=8.5, loc="left")
    r = np.corrcoef(x, y)[0, 1]; th = np.arccos(r)
    seta(a2, (0, 0), (np.cos(th), np.sin(th)), CATEG[0])
    seta(a2, (0, 0), (np.cos(th), 0), CATEG[1])
    a2.plot([np.cos(th), np.cos(th)], [0, np.sin(th)], color=CATEG[2], lw=1.6, ls="--")
    a2.plot([np.cos(th) - 0.05, np.cos(th) - 0.05, np.cos(th)], [0, 0.05, 0.05], color="#666", lw=0.7)
    a2.text(np.cos(th) / 2 - 0.05, np.sin(th) / 2 + 0.05, r"$\mathbf{y}-\bar y\mathbf{1}$", fontsize=8, ha="right")
    a2.text(np.cos(th) / 2, -0.09, r"$\hat{\mathbf{y}}-\bar y\mathbf{1}$", fontsize=8, ha="center", va="top")
    a2.text(np.cos(th) + 0.04, np.sin(th) / 2, "resíduo", fontsize=8, va="center")
    a2.text(0.16, 0.05, f"{np.degrees(th):.1f}°", fontsize=7.5)
    a2.text(0.0, 1.07, f"$R^2=\\cos^2\\theta={r**2:.3f}$", fontsize=8)
    a2.set_xlim(-0.1, 1.0); a2.set_ylim(-0.25, 1.15); a2.set_aspect("equal"); a2.axis("off")
    a2.set_title("(b) geometria (vetores normalizados)", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "salario_regressao")


# ---------------------------------------------------------------------------
# 7. Média móvel: tendência e sazonalidade do consumo de gás
# ---------------------------------------------------------------------------
def fig_media_movel():
    g = pd.read_csv(DADOS / "ukgas.csv")
    y = np.log(g["consumo"].to_numpy()); n = len(y)
    t = g["ano"].to_numpy() + (g["trimestre"].to_numpy() - 1) / 4
    tend = np.convolve(y, np.ones(4) / 4, mode="valid")
    tc = t[:n - 3] + 1.5 / 4
    desv = y[1:n - 2] - tend
    tri = g["trimestre"].to_numpy()[1:n - 2]
    saz = np.array([desv[tri == q].mean() for q in range(1, 5)]); saz -= saz.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3), gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(t, y, color=CATEG[0], lw=1, label="log(consumo)")
    a1.plot(tc, tend, color=CATEG[1], lw=2, label="média móvel ($\\mathbf{M}\\mathbf{y}$)")
    a1.set_xlabel("ano"); a1.set_ylabel("log(milhões de termias)")
    a1.legend(fontsize=7, loc="upper left")
    a1.set_title("(a) série e tendência", fontsize=8.5, loc="left")
    a2.bar(range(1, 5), saz, color=CATEG[2], width=0.6)
    a2.axhline(0, color="#999", lw=0.6)
    a2.set_xticks(range(1, 5), ["T1", "T2", "T3", "T4"])
    a2.set_ylabel("efeito sazonal (log)")
    a2.set_title("(b) padrão no núcleo de $\\mathbf{M}$", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "media_movel")


# ---------------------------------------------------------------------------
# 8. Contrastes polinomiais da resposta ao nitrogênio
# ---------------------------------------------------------------------------
def fig_contrastes():
    d = pd.read_csv(DADOS / "yates_oats.csv")
    doses = np.sort(d["nitro"].unique())
    m = d.groupby("nitro")["yield"].mean().reindex(doses).to_numpy()
    C = np.array([[-3, -1, 1, 3], [1, -1, -1, 1], [-1, 3, -3, 1]]).T
    sq = 18 * (C.T @ m) ** 2 / (C ** 2).sum(axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.plot(doses, m, "o", color=CATEG[0], ms=6, label="médias por dose", zorder=3)
    xs = np.linspace(0, 0.6, 100)
    for grau, cor, ls in ((1, CATEG[1], "-"), (2, CATEG[2], "--")):
        c = np.polyfit(doses, m, grau)
        a1.plot(xs, np.polyval(c, xs), color=cor, ls=ls, lw=1.6, label=f"ajuste de grau {grau}")
    a1.set_xlabel("dose de nitrogênio (cwt/acre)"); a1.set_ylabel("produção média")
    a1.legend(fontsize=7)
    a1.set_title("(a) curva de resposta", fontsize=8.5, loc="left")
    nomes = ["linear", "quadrático", "cúbico"]
    a2.barh(nomes[::-1], sq[::-1], color=[CATEG[3], CATEG[2], CATEG[1]], height=0.55)
    for i, v in enumerate(sq[::-1]):
        a2.text(v + 300, i, f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."), va="center", fontsize=7)
    a2.set_xlim(0, 26000); a2.set_xlabel("soma de quadrados")
    a2.set_title("(b) SQ$_{trat}$ = soma de 3 projeções", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "aveia_contrastes")


# ---------------------------------------------------------------------------
# 9. Dígitos: matriz de confusão
# ---------------------------------------------------------------------------
def fig_confusao():
    d = pd.read_csv(DADOS / "digitos.csv")
    X = d.drop(columns="digito").to_numpy(float); r = d["digito"].to_numpy()
    C = np.vstack([X[:1200][r[:1200] == k].mean(0) for k in range(10)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Xt = X[1200:] / np.linalg.norm(X[1200:], axis=1, keepdims=True)
    pred = (Xt @ C.T).argmax(1)
    M = np.zeros((10, 10), int)
    for a, b in zip(r[1200:], pred):
        M[a, b] += 1
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(M, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 30 else "#222")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("previsto"); ax.set_ylabel("verdadeiro"); ax.grid(False)
    ax.tick_params(length=0, labelsize=7)
    salvar(fig, CAP, "digitos_confusao")


# ---------------------------------------------------------------------------
# 10. Rede linear: trajetória do treinamento até a solução de MQO
# ---------------------------------------------------------------------------
def fig_rede_linear():
    rng = np.random.default_rng(42)
    n, p, h = 500, 5, 8
    X = rng.normal(size=(n, p)); beta = np.array([2, -1, 0.5, 0, 3])
    y = X @ beta + rng.normal(scale=0.5, size=n)
    mqo = np.linalg.lstsq(X, y, rcond=None)[0]
    W1 = rng.normal(scale=0.3, size=(h, p)); w2 = rng.normal(scale=0.3, size=(1, h))
    hist, perda = [], []
    for it in range(3000):
        W = (w2 @ W1).ravel(); hist.append(W); res = X @ W - y
        perda.append(0.5 * np.mean(res ** 2))
        g = (X.T @ res / n)[None, :]
        w2, W1 = w2 - 0.02 * (g @ W1.T), W1 - 0.02 * (w2.T @ g)
    hist = np.array(hist)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.2))
    p_mqo = 0.5 * np.mean((X @ mqo - y) ** 2)
    it = np.arange(1, len(perda) + 1)
    a1.loglog(it, np.array(perda) - p_mqo, color=CATEG[0])
    a1.set_xlabel("iteração"); a1.set_ylabel("perda $-$ perda mínima (MQO)")
    a1.set_title("(a) excesso de perda", fontsize=8.5, loc="left")
    for j in range(p):
        a2.semilogx(it, hist[:, j], color=CATEG[j % 5], lw=1.4)
        a2.axhline(mqo[j], color=CATEG[j % 5], ls=":", lw=0.9)
        a2.text(3600, mqo[j], f"$\\hat\\beta_{j+1}$", fontsize=7, va="center", color="#333")
    a2.set_xlim(1, 9000); a2.set_xlabel("iteração")
    a2.set_ylabel(r"entradas de $\mathbf{w}_2\mathbf{W}_1$")
    a2.set_title("(b) produto das camadas → MQO", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "rede_linear")


# ---------------------------------------------------------------------------
# 11. Aproximação em dimensão infinita: projeções de |t| em P_d (Legendre)
# ---------------------------------------------------------------------------
def fig_legendre():
    from numpy.polynomial import legendre as L
    t = np.linspace(-1, 1, 801); f = np.abs(t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.2))
    a1.plot(t, f, color="#222", lw=1.8, label="$f(t)=|t|$")
    erros = []
    graus = range(0, 21, 2)
    for d in graus:
        # coeficientes de Legendre por quadratura de Gauss
        xq, wq = L.leggauss(200)
        c = np.array([(2 * k + 1) / 2 * np.sum(wq * np.abs(xq) * L.legval(xq, np.eye(k + 1)[k])) for k in range(d + 1)])
        pd_ = L.legval(t, c)
        erros.append(np.sqrt(np.sum(wq * (np.abs(xq) - L.legval(xq, c)) ** 2)))
        if d in (2, 4, 10):
            a1.plot(t, pd_, lw=1.3, label=f"$\\Pi_{{\\mathcal{{P}}_{{{d}}}}}f$",
                    color={2: CATEG[1], 4: CATEG[2], 10: CATEG[0]}[d])
    a1.legend(fontsize=7, loc="upper center")
    a1.set_title("(a) projeções ortogonais em $\\mathcal{P}_d$", fontsize=8.5, loc="left")
    a2.semilogy(list(graus), erros, "o-", color=CATEG[0], ms=4)
    a2.set_xlabel("grau $d$"); a2.set_ylabel("$\\|f-\\Pi_{\\mathcal{P}_d}f\\|$")
    a2.set_title("(b) o erro tende a zero, mas nunca se anula", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "legendre_abs")


# ---------------------------------------------------------------------------
# 12. Índices de preços: contribuições e substituição
# ---------------------------------------------------------------------------
def fig_precos():
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))
    bens = ["arroz", "feijão", "carne", "leite", "pão"]
    p0 = np.array([5.20, 8.90, 38.50, 4.60, 12.00]); p1 = np.array([6.10, 8.10, 44.90, 5.30, 13.20])
    q0 = np.array([10, 4, 6, 20, 8.0]); q1 = np.array([11, 5, 4.5, 19, 8])
    w = p0 * q0 / (p0 @ q0); r = 100 * (p1 / p0 - 1); L = 100 * ((p1 @ q0) / (p0 @ q0) - 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6))
    x0 = np.r_[0, np.cumsum(w)[:-1]]
    for k in range(5):
        a1.bar(x0[k], r[k], width=w[k], align="edge", color=CATEG[k % 5], alpha=0.8, edgecolor="white")
        a1.text(x0[k] + w[k] / 2, r[k] + (1 if r[k] > 0 else -3), bens[k], ha="center", fontsize=6.5)
    a1.axhline(L, color=COR["cinza"], ls="--", lw=1)
    a1.text(0.98, -11, f"tracejado: Laspeyres = {L:.1f}%".replace(".", ","), ha="right", fontsize=6.5, color=COR["cinza"])
    a1.set_ylim(-13, 22)
    a1.axhline(0, color="black", lw=0.6)
    a1.set_xlim(0, 1); a1.set_xlabel("peso no dispêndio do período base"); a1.set_ylabel("variação do preço (%)")
    a1.set_title("(a) área = contribuição à inflação", fontsize=8.5, loc="left")
    a1.xaxis.set_major_formatter(fmt); a1.yaxis.set_major_formatter(fmt)
    dq = 100 * (q1 / q0 - 1)
    a2.axhline(0, color=COR["cinza"], lw=0.6); a2.axvline(0, color=COR["cinza"], lw=0.6)
    for k in range(5):
        a2.plot(r[k], dq[k], "o", ms=5, color=CATEG[k % 5])
        a2.annotate(bens[k], (r[k], dq[k]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    a2.set_xlabel("variação do preço (%)"); a2.set_ylabel("variação da quantidade (%)")
    a2.set_title("(b) substituição", fontsize=8.5, loc="left")
    a2.xaxis.set_major_formatter(fmt); a2.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); salvar(fig, CAP, "precos")


if __name__ == "__main__":
    fig_correlacao_cosseno()
    fig_aveia()
    fig_digitos()
    fig_camadas()
    fig_bolas()
    fig_salario()
    fig_media_movel()
    fig_contrastes()
    fig_confusao()
    fig_rede_linear()
    fig_legendre()
    fig_precos()
