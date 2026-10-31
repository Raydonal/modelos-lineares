"""Cap. 10: Deriva de embeddings detectada pelo T^2 de Hotelling."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.austen import embeddings_austen  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

vocab, _, E = embeddings_austen()
ind = {w: i for i, w in enumerate(vocab)}
raiz = Path(__file__).resolve().parents[2]
d = pd.read_csv(raiz / "dados" / "austen.csv", keep_default_na=False)


def blocos(livro, L=200):                     # médias de 200 palavras
    txt = " ".join(d["texto"][d["livro"] == livro]).lower()
    ids = [ind[w] for w in re.findall(r"[a-z]+", txt) if w in ind]
    m = len(ids) // L
    return E[ids[:m * L]].reshape(m, L, -1).mean(1)


pp = blocos("Pride & Prejudice"); h = len(pp) // 2
ref = pp[:h]; mu = ref.mean(0); p = 10
V = np.linalg.svd(ref - mu, full_matrices=False)[2][:p].T  # PCA


def t2(X1, X2):
    Z1, Z2 = (X1 - mu) @ V, (X2 - mu) @ V
    n1, n2 = len(Z1), len(Z2); g = n1 + n2 - 2
    dm = Z1.mean(0) - Z2.mean(0)
    Sp = ((n1 - 1) * np.cov(Z1.T) + (n2 - 1) * np.cov(Z2.T)) / g
    T2 = n1 * n2 / (n1 + n2) * dm @ np.linalg.solve(Sp, dm)
    Fst = T2 * (g - p + 1) / (p * g)
    return T2, Fst, f.sf(Fst, p, g - p + 1)


lote = {"PP_2a_metade": pp[h:h + 30], "Emma": blocos("Emma")[:30],
        "Persuasion": blocos("Persuasion")[:30],
        "Sense": blocos("Sense & Sensibility")[:30]}
for nome, X in lote.items():
    T2, Fst, pv = t2(ref, X)
    print(f"{nome:13s} T2 = {T2:7.2f}  F = {Fst:6.3f}  p = {pv:.4g}")
# Calibração: 500 partições ao acaso da referência
g = Gerador(41); k = h // 2; rej = 0
for _ in range(500):
    o = np.argsort(g.unif(h))
    rej += t2(ref[o[:k]], ref[o[k:]])[2] < 0.05
print("n_referência =", h, " falsos positivos =", rej / 500)
