"""Cap. 14: Figuras de dados do capítulo (em figuras/cap14/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FuncFormatter
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import (  # noqa: E402
    CATEG, COR, DADOS, aplicar_estilo, salvar)
from comum.gerador import Gerador  # noqa: E402
from comum.multiplas import bh, q_dunnett  # noqa: E402

aplicar_estilo()
CAP = "cap14"
LARG = 5.3
_fmt = FuncFormatter(
    lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. Poder do teste F
def fig_poder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3))
    lam = np.linspace(0.01, 30, 121)
    for j, q in enumerate([1, 3, 6, 12]):
        cr = stats.f.ppf(0.95, q, 30)
        a1.plot(lam, stats.ncf.sf(cr, q, 30, lam), color=CATEG[j],
                label=f"$q={q}$")
    a1.axhline(0.05, color=COR["cinza"], lw=0.6, ls=":")
    a1.set_xlabel(r"não centralidade $\lambda$")
    a1.set_ylabel("poder")
    a1.legend(fontsize=7, loc="lower right")
    rr = np.arange(4, 41)
    for j, dl in enumerate([0.5, 1.0, 1.5]):
        pw = [stats.ncf.sf(stats.f.ppf(0.95, 1, 2 * r - 2), 1,
                           2 * r - 2, r * dl**2 / 2) for r in rr]
        a2.plot(rr, pw, color=CATEG[j],
                label=rf"$\delta/\sigma={dl:g}$".replace(".", "{,}"))
    a2.axhline(0.8, color=COR["cinza"], lw=0.6, ls=":")
    a2.set_xlabel("unidades por grupo, $n_g$")
    a2.legend(fontsize=7, loc="lower right")
    for a, t in ((a1, "(a)"), (a2, "(b)")):
        virgula(a); a.set_title(t, loc="left", fontsize=8)
    fig.tight_layout()
    salvar(fig, CAP, "poder")


# 2. Elipse, retângulos e sombras
def fig_regioes():
    w = pd.read_csv(DADOS / "wage1.csv")
    y = np.log(w.wage.to_numpy()); n = len(y)
    X = np.column_stack([np.ones(n), w.educ, w.female, w.exper,
                         w.tenure]).astype(float)
    A = np.linalg.inv(X.T @ X); b = A @ X.T @ y
    nu = n - X.shape[1]; s2 = np.sum((y - X @ b) ** 2) / nu
    j = [3, 4]                            # exper e tenure (x 100)
    C = 1e4 * s2 * A[np.ix_(j, j)]; c = 100 * b[j]
    F2 = stats.f.ppf(0.95, 2, nu)
    val, vec = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    se = np.sqrt(np.diag(C))
    mt = stats.t.ppf(0.975, nu); mb = stats.t.ppf(1 - 0.0125, nu)
    ms = np.sqrt(2 * F2)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.add_patch(Ellipse(c, 2 * np.sqrt(2 * F2 * val[1]),
                         2 * np.sqrt(2 * F2 * val[0]), angle=ang,
                         fc=CATEG[0] + "33", ec=CATEG[0], lw=1.2))
    for mm, cor, nm, ls in ((mt, CATEG[2], "$t$", ":"),
                            (mb, CATEG[1], "Bonferroni", "--"),
                            (ms, COR["cinza"], "Scheffé", "-")):
        ax.add_patch(Rectangle(c - mm * se, *(2 * mm * se), fc="none",
                               ec=cor, lw=0.9, ls=ls))
        ax.plot([], [], color=cor, ls=ls, lw=0.9, label=nm)
    ax.plot(*c, "o", color=CATEG[0], ms=3)
    ax.set_xlim(c[0] - 1.15 * ms * se[0], c[0] + 1.7 * ms * se[0])
    ax.set_ylim(c[1] - 1.15 * ms * se[1], c[1] + 1.15 * ms * se[1])
    ax.set_xlabel(r"$100\,\beta_{\mathrm{exper}}$")
    ax.set_ylabel(r"$100\,\beta_{\mathrm{tenure}}$")
    ax.legend(fontsize=7, loc="upper right")
    virgula(ax)
    salvar(fig, CAP, "regioes")


# 3. Multiplicadores contra o número de grupos
def fig_multiplicadores():
    aa = np.array([3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
    t, bo, tu, du, sc = [], [], [], [], []
    for a in aa:
        nu = 4 * a; m = a * (a - 1) / 2
        t.append(stats.t.ppf(0.975, nu))
        bo.append(stats.t.ppf(1 - 0.025 / m, nu))
        tu.append(stats.studentized_range.ppf(0.95, a, nu)
                  / np.sqrt(2))
        du.append(q_dunnett(int(a) - 1, nu))
        sc.append(np.sqrt((a - 1) * stats.f.ppf(0.95, a - 1, nu)))
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    for v, nm, c in ((sc, "Scheffé", CATEG[4]), (bo, "Bonferroni",
                                                 CATEG[1]),
                     (tu, "Tukey", CATEG[0]),
                     (du, "Dunnett", CATEG[2]),
                     (t, "$t$ sem ajuste", COR["cinza"])):
        ax.plot(aa, v, color=c, marker="o", ms=2)
        dy = {"Bonferroni": 0.13, "Tukey": -0.13}.get(nm, 0)
        ax.text(30.6, v[-1] + dy, nm, fontsize=7, color=c,
                va="center")
    ax.set_xlim(2, 37)
    ax.set_xlabel("número de grupos, $a$")
    ax.set_ylabel("multiplicador do erro-padrão")
    virgula(ax)
    salvar(fig, CAP, "multiplicadores")


# 4. Salários: formas, falta de ajuste e faixas
def fig_salarios():
    w = pd.read_csv(DADOS / "wage1.csv")
    y = np.log(w.wage.to_numpy()); e = w.educ.to_numpy(float)
    n = len(y); g = w.assign(ly=y).groupby("educ").ly
    ej = g.mean().index.to_numpy(float); mj = g.mean().to_numpy()
    nj = g.size().to_numpy()
    grade = np.linspace(0, 18, 181)

    def ajusta(f):
        X = f(e); return np.linalg.solve(X.T @ X, X.T @ y)

    fs = {"reta": lambda v: np.column_stack([np.ones(len(v)), v]),
          "parábola": lambda v: np.column_stack(
              [np.ones(len(v)), v, v**2]),
          "quebrada em 12": lambda v: np.column_stack(
              [np.ones(len(v)), v, np.maximum(v - 12, 0)])}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.scatter(ej, mj, s=nj * 0.6, color=COR["cinza"], alpha=0.7,
               lw=0, zorder=3)
    for j, (nm, f) in enumerate(fs.items()):
        a1.plot(grade, f(grade) @ ajusta(f), color=CATEG[j], label=nm)
    a1.set_xlabel("anos de escolaridade")
    a1.set_ylabel("log do salário-hora")
    a1.legend(fontsize=7, loc="upper left")
    f = fs["quebrada em 12"]; X = f(e); A = np.linalg.inv(X.T @ X)
    b = A @ X.T @ y; s2 = np.sum((y - X @ b) ** 2) / (n - 3)
    G = f(grade); ft = G @ b; h = ((G @ A) * G).sum(axis=1)
    tq = stats.t.ppf(0.975, n - 3)
    mw = np.sqrt(3 * stats.f.ppf(0.95, 3, n - 3))
    a2.scatter(e + 0.15 * np.sin(np.arange(n)), y, s=2,
               color=COR["cinza"], alpha=0.35, lw=0)
    a2.fill_between(grade, ft - tq * np.sqrt(s2 * (1 + h)),
                    ft + tq * np.sqrt(s2 * (1 + h)),
                    color=CATEG[0], alpha=0.13, lw=0,
                    label="predição")
    a2.fill_between(grade, ft - mw * np.sqrt(s2 * h),
                    ft + mw * np.sqrt(s2 * h), color=CATEG[1],
                    alpha=0.45, lw=0, label="Working--Hotelling")
    a2.plot(grade, ft, color=CATEG[0], lw=1)
    a2.set_xlabel("anos de escolaridade")
    a2.legend(fontsize=7, loc="lower right")
    for a, t in ((a1, "(a)"), (a2, "(b)")):
        virgula(a); a.set_title(t, loc="left", fontsize=8)
    fig.tight_layout()
    salvar(fig, CAP, "salarios")


# 5. Aveia: resposta à dose e intervalos simultâneos
def fig_aveia():
    a = pd.read_csv(DADOS / "yates_oats.csv")
    md = a.groupby("nitro")["yield"].mean()
    d = md.index.to_numpy(float); md = md.to_numpy()
    qm = 177.083; ep1 = np.sqrt(qm / 18); ep = np.sqrt(2 * qm / 18)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3),
                                 gridspec_kw={"width_ratios": [1, 1]})
    a1.errorbar(d, md, yerr=2.014 * ep1, fmt="o", color=CATEG[0],
                ms=3, capsize=2, zorder=3, label="médias")
    dd = np.linspace(0, 0.6, 61)
    p1 = np.polyfit(d, md, 1); p2 = np.polyfit(d, md, 2)
    a1.plot(dd, np.polyval(p1, dd), color=CATEG[1], label="reta")
    a1.plot(dd, np.polyval(p2, dd), color=CATEG[2], ls="--",
            label="parábola")
    a1.set_xlabel("dose de nitrogênio")
    a1.set_ylabel("produção média")
    a1.legend(fontsize=7, loc="lower right")
    mult = {"$t$": (stats.t.ppf(0.975, 45), COR["cinza"]),
            "Dunnett": (q_dunnett(3, 45), CATEG[0])}
    for k in (1, 2, 3):
        for j, (nm, (mm, c)) in enumerate(mult.items()):
            yy = k + 0.13 * (0.5 - j)
            a2.errorbar(md[k] - md[0], yy, xerr=mm * ep, fmt="o",
                        ms=2.5, color=c, capsize=2,
                        label=nm if k == 1 else None)
    a2.axvline(0, color=COR["cinza"], lw=0.6, ls=":")
    a2.set_yticks([1, 2, 3])
    a2.set_yticklabels(["0,2", "0,4", "0,6"])
    a2.set_ylabel("dose")
    a2.set_xlabel("diferença para a dose zero")
    a2.legend(fontsize=7, loc="lower right")
    for ax, t in ((a1, "(a)"), (a2, "(b)")):
        ax.set_title(t, loc="left", fontsize=8)
    virgula(a1); virgula(a2, "x")
    fig.tight_layout()
    salvar(fig, CAP, "aveia")


# 6. NFL: p ordenados, Bonferroni e BH
def fig_nfl():
    nfl = pd.read_csv(DADOS / "nfl.csv"); s = nfl[nfl.ano == 1981]
    tm = sorted(set(s.mandante) | set(s.visitante)); k = len(tm)
    n = len(s)
    y = (s.pontos_mandante - s.pontos_visitante).to_numpy(float)
    idx = {t: i for i, t in enumerate(tm)}
    X = np.zeros((n, k + 1)); X[:, k] = 1
    X[np.arange(n), s.mandante.map(idx)] = 1
    X[np.arange(n), s.visitante.map(idx)] = -1
    G = np.linalg.pinv(X.T @ X); b = G @ X.T @ y
    nu = n - 28; s2 = np.sum((y - X @ b) ** 2) / nu
    i1, i2 = np.triu_indices(k, 1); m = len(i1)
    L = np.zeros((m, k + 1))
    L[np.arange(m), i1] = 1; L[np.arange(m), i2] = -1
    se2 = ((L @ G) * L).sum(axis=1)
    tt = (L @ b) / np.sqrt(s2 * se2)
    pv = np.sort(2 * stats.t.sf(np.abs(tt), nu))
    R = bh(pv, 0.05); r = np.arange(1, m + 1)
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    lim = 90
    ax.plot(r[:lim], pv[:lim], ".", ms=3, color=CATEG[0], label="$p$")
    ax.plot(r[:lim], 0.05 * r[:lim] / m, color=CATEG[1],
            label=r"$\alpha k/m$ (BH)")
    ax.axhline(0.05 / m, color=CATEG[2], ls="--", lw=0.9,
               label="Bonferroni")
    ax.axvline(R, color=COR["cinza"], lw=0.6, ls=":")
    ax.text(R + 1, 0.0105, f"$R={R}$", fontsize=7, color=COR["cinza"])
    ax.set_ylim(0, 0.012); ax.set_xlabel("posição $k$ do par")
    ax.set_ylabel("valor-$p$ ordenado")
    ax.legend(fontsize=7, loc="upper left")
    virgula(ax)
    salvar(fig, CAP, "nfl")


# 7. Fronteira
def fig_fronteira():
    a, n = 6, 3; N = 200000; g = Gerador(1987)
    sqb = stats.chi2.ppf(g.unif(N), a - 1)
    sqd = stats.chi2.ppf(g.unif(N), a * (n - 1))
    Fv = (sqb / (a - 1)) / (sqd / (a * (n - 1)))
    L = np.where(Fv <= a / (a - 1), 0.0,
                 a * n * np.log((sqb + sqd) / (a * n))
                 - a * (n - 1) * np.log(sqd / (a * (n - 1)))
                 - a * np.log(sqb / a))
    xx = np.linspace(0.001, 6, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3))
    emp = np.searchsorted(np.sort(L), xx, side="right") / N
    a1.plot(xx, emp, color=CATEG[0], label="simulada")
    a1.plot(xx, 0.5 + 0.5 * stats.chi2.cdf(xx, 1), color=CATEG[1],
            ls="--",
            label=r"$\frac{1}{2}\chi^2_0+\frac{1}{2}\chi^2_1$")
    a1.plot(xx, stats.chi2.cdf(xx, 1), color=COR["cinza"], ls=":",
            label=r"$\chi^2_1$")
    a1.set_xlabel("$x$"); a1.set_ylabel(r"$\Pr(\mathrm{RV}\leq x)$")
    a1.legend(fontsize=7, loc="lower right")
    av = pd.read_csv(DADOS / "yates_oats.csv")
    y = av["yield"].to_numpy(float); m = len(y); link = 0.2012
    bl = sorted(av.block.unique()); ds = sorted(av.nitro.unique())
    gn = sorted(av.gen.unique())
    X = np.column_stack(
        [np.ones(m)]
        + [(av.block == v).to_numpy(float) for v in bl[1:]]
        + [(av.nitro == v).to_numpy(float) for v in ds[1:]]
        + [(av.gen == v).to_numpy(float) for v in gn[1:]])
    cx = (av.col.to_numpy() - 1) * 44 * link
    cy = (av.row.to_numpy() - 1) * 28.4 * link
    D = np.hypot(cx[:, None] - cx, cy[:, None] - cy)
    pre = []
    for f in [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 29, 35, 45, 60]:
        R = np.eye(m) if f == 0 else np.exp(-D / f)
        Ri = np.linalg.inv(R); RiX = Ri @ X
        A = Ri - RiX @ np.linalg.solve(X.T @ RiX, RiX.T)
        pre.append(((A + A.T) / 2, np.linalg.slogdet(R)[1]))

    def perf(Y):
        return np.column_stack(
            [-m / 2 * np.log(((Y @ A) * Y).sum(axis=1) / m) - ld / 2
             for A, ld in pre])

    N = 20000; g = Gerador(2014)
    Lz = perf(g.matriz_normal(N, m))
    rv0 = 2 * (Lz.max(axis=1) - Lz[:, 0])
    ob = perf(y[None, :])[0]; rvobs = 2 * (ob.max() - ob[0])
    xx = np.linspace(0.0, 13, 400)
    cau = 1 - np.searchsorted(np.sort(rv0), xx, side="right") / N
    a2.plot(xx, cau, color=CATEG[0], label="simulada")
    a2.plot(xx[1:], 0.5 * stats.chi2.sf(xx[1:], 1), color=CATEG[1],
            ls="--", label="mistura")
    a2.plot(xx[1:], stats.chi2.sf(xx[1:], 1), color=COR["cinza"],
            ls=":", label=r"$\chi^2_1$")
    a2.axvline(rvobs, color=COR["destaque"], lw=0.8)
    a2.text(rvobs - 0.2, 0.25, "obs.", fontsize=7, ha="right",
            color=COR["destaque"])
    a2.set_yscale("log"); a2.set_ylim(5e-5, 1.2)
    a2.set_xlabel("$x$"); a2.set_ylabel(r"$\Pr(\mathrm{RV}>x)$")
    a2.legend(fontsize=7, loc="lower left")
    for ax, t in ((a1, "(a)"), (a2, "(b)")):
        ax.set_title(t, loc="left", fontsize=8)
    virgula(a1); virgula(a2, "x")
    fig.tight_layout()
    salvar(fig, CAP, "fronteira")


# 8. Dígitos
def fig_digitos():
    dig = pd.read_csv(DADOS / "digitos.csv")
    P = dig.iloc[:, 1:].to_numpy(float) / 16
    rot = dig.digito.to_numpy()
    tr = np.arange(1200); te = np.arange(1200, len(P))
    Xc = P - P[tr].mean(axis=0); Xt = Xc[tr]
    Y = (rot[tr, None] == np.arange(10)) * 1.0
    ev = np.linalg.eigh(Xt.T @ Xt)[1][:, ::-1]

    def centr(Sc):
        C = np.vstack([Sc[tr][rot[tr] == j].mean(axis=0)
                       for j in range(10)])
        return ((Sc[te, None, :] - C[None, :, :]) ** 2).sum(
            axis=2).argmin(axis=1)

    def lin(Sc, lam=0.0):
        St = Sc[tr]
        B = np.linalg.solve(St.T @ St + lam * np.eye(St.shape[1]),
                            St.T @ (Y - Y.mean(axis=0)))
        return (Sc[te] @ B).argmax(axis=1)

    def knn(k):
        out = []
        for i in te:
            d = ((P[tr] - P[i]) ** 2).sum(axis=1)
            v = rot[tr][np.argsort(d, kind="stable")[:k]]
            out.append(np.bincount(v, minlength=10).argmax())
        return np.array(out)

    C = {"cent": centr(Xc)}
    for k in (10, 20, 40):
        C[f"cent{k}"] = centr(Xc @ ev[:, :k])
    for k in (10, 20, 30, 40, 50):
        C[f"lin{k}"] = lin(Xc @ ev[:, :k])
    for l in (0.01, 0.1, 1, 10, 100, 1000):
        C[f"ridge{l:g}"] = lin(Xc, l)
    for k in (1, 3, 5, 7, 9):
        C[f"knn{k}"] = knn(k)
    nomes = list(C)
    Ac = np.column_stack([(C[k] == rot[te]) * 1.0 for k in nomes])
    o = np.argsort(Gerador(597).unif(597)); val = o[:300]
    b0 = nomes.index("lin30")
    nm2 = [k for k in nomes if k != "lin30"]
    D = np.delete(Ac[val], b0, axis=1) - Ac[val][:, [b0]]
    nd = np.sqrt((D**2).sum(axis=0)); nd[nd == 0] = 1
    tt = D.sum(axis=0) / nd
    g = Gerador(1998); N = 20000
    E = (2.0 * (g.unif(N * 300) < 0.5) - 1).reshape(N, 300)
    mx = np.abs(E @ (D / nd)).max(axis=1)
    cmax = np.sort(mx)[int(0.95 * N) - 1]
    ord_ = np.argsort(tt)
    fig, ax = plt.subplots(figsize=(LARG, 2.3))
    cores = [CATEG[1] if abs(tt[i]) > cmax else COR["cinza"]
             for i in ord_]
    ax.bar(range(len(tt)), tt[ord_], color=cores, width=0.7)
    ax.set_xticks(range(len(tt)))
    ax.set_xticklabels([nm2[i] for i in ord_], rotation=60,
                       ha="right", fontsize=6)
    bo = stats.norm.ppf(1 - 0.025 / 19)
    for v, c, ls, nm in ((1.96, COR["cinza"], ":", r"$1{,}96$"),
                         (cmax, CATEG[1], "--", "max-$t$"),
                         (bo, CATEG[2], "-.", "Bonferroni")):
        ax.axhline(v, color=c, ls=ls, lw=0.8)
        ax.axhline(-v, color=c, ls=ls, lw=0.8)
        ax.text(-0.4, v + 0.08, nm, fontsize=6.5, color=c)
    ax.set_ylabel("estatística de McNemar")
    ax.grid(axis="x", visible=False)
    virgula(ax, "y")
    salvar(fig, CAP, "digitos")


if __name__ == "__main__":
    fig_poder(); fig_regioes(); fig_multiplicadores(); fig_salarios()
    fig_aveia(); fig_nfl(); fig_fronteira(); fig_digitos()
