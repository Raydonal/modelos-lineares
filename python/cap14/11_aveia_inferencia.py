"""Cap. 14: Aveia: aditividade, forma da dose e Dunnett."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.multiplas import q_dunnett  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
a = a.sort_values(["block", "gen", "nitro"]).reset_index(drop=True)
y = a["yield"].to_numpy(float); n = len(y)


def um(f):
    return pd.get_dummies(f).to_numpy(float)


Bl = um(a.block); Gn = um(a.gen); Do = um(a.nitro)
Pc = um(a.block + a.gen); Ce = um(a.gen + a.nitro.astype(str))


def sqr(*cols):
    Z = np.column_stack(cols)
    e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return e @ e


# Estrato das subparcelas: parcelas + doses + variedade x dose
s_pd = sqr(Pc, Do); s_int = sqr(Pc, Ce)
s_p = sqr(Pc)
qmsub = s_int / 45
print(f"SQ erro sub {s_int:.2f} QM {qmsub:.2f} gl 45")
Fi = ((s_pd - s_int) / 6) / qmsub
print(f"interação: SQ {s_pd - s_int:.1f} F(6,45) {Fi:.3f}"
      f" p {stats.f.sf(Fi, 6, 45):.3f}")
Fd = ((s_p - s_pd) / 3) / qmsub
print(f"dose: SQ {s_p - s_pd:.1f} F(3,45) {Fd:.2f}"
      f" p {stats.f.sf(Fd, 3, 45):.1e}")
# Mesma hipótese com o erro ingênuo (72 - 17 = 55 gl)
s_b = sqr(Bl, Gn, Do); s_c = sqr(Bl, Ce)
Fn = ((s_b - s_c) / 6) / (s_c / 55)
print(f"interação ingênua: QM {s_c / 55:.1f} F(6,55) {Fn:.3f}"
      f" p {stats.f.sf(Fn, 6, 55):.3f}")
# Forma da dose dentro das parcelas: reta contra as 4 médias
d = a.nitro.to_numpy(float); s_lin = sqr(Pc, d)
Fl = ((s_lin - s_pd) / 2) / qmsub
print(f"reta: SQ lin {s_p - s_lin:.1f} falta {s_lin - s_pd:.1f}"
      f" F(2,45) {Fl:.3f} p {stats.f.sf(Fl, 2, 45):.3f}")
s_q = sqr(Pc, d, d**2)
Fq = ((s_q - s_pd) / 1) / qmsub
print(f"parábola: falta {s_q - s_pd:.1f} F(1,45) {Fq:.3f}"
      f" p {stats.f.sf(Fq, 1, 45):.3f}")
Fc = ((s_lin - s_q) / 1) / qmsub
print(f"termo quadrático: SQ {s_lin - s_q:.1f} F(1,45) {Fc:.3f}"
      f" p {stats.f.sf(Fc, 1, 45):.3f}")
# Tukey (um grau) nas médias de parcela: bloco x variedade
Y = a.pivot_table(index="block", columns="gen",
                  values="yield", aggfunc="mean").to_numpy()
A = Y.mean(axis=1) - Y.mean(); B = Y.mean(axis=0) - Y.mean()
E = Y - np.add.outer(A, B) - Y.mean()
D = np.sum(np.outer(A, B) * Y) ** 2 / (A @ A * (B @ B))
gl = 5 * 2 - 1
Ft = D / ((np.sum(E**2) - D) / gl)
print(f"Tukey: SQRes {np.sum(E**2):.1f} D {D:.1f} F(1,9) {Ft:.3f}"
      f" p {stats.f.sf(Ft, 1, gl):.3f}")
Fg = (4 * 6 * (B @ B) / 2) / (4 * np.sum(E**2) / 10)
print(f"variedade: F(2,10) {Fg:.3f} p {stats.f.sf(Fg, 2, 10):.3f}"
      f" (QM parcelas {4 * np.sum(E**2) / 10:.1f})")
# Doses contra a dose zero e todos os pares (r = 18)
md = a.groupby("nitro")["yield"].mean()
vd = md.index.to_numpy(); md = md.to_numpy()
ep = np.sqrt(2 * qmsub / 18)
print(f"médias {md[0]:.2f} {md[1]:.2f} {md[2]:.2f} {md[3]:.2f}"
      f" ep {ep:.3f}")
mt = stats.t.ppf(0.975, 45); mb = stats.t.ppf(1 - 0.05 / 6, 45)
md3 = q_dunnett(3, 45)
mtk = stats.studentized_range.ppf(0.95, 4, 45) / np.sqrt(2)
msc = np.sqrt(3 * stats.f.ppf(0.95, 3, 45))
print(f"mult.: t {mt:.3f} Bonf {mb:.3f} Dunn {md3:.3f}"
      f" Tukey {mtk:.3f} Sch {msc:.3f}")
for k in (1, 2, 3):
    print(f"{vd[k]:.1f} - 0: {md[k] - md[0]:.2f} +- {md3 * ep:.2f}"
          " (Dunnett)")
dfp = np.diff(md)
print(f"pares vizinhos {dfp[0]:.2f} {dfp[1]:.2f} {dfp[2]:.2f},"
      f" Tukey +- {mtk * ep:.2f}")
