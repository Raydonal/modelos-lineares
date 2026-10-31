"""Cap. 6: Embeddings de palavras pela SVD da matriz PPMI de coocorrências (romances de Jane Austen)."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "austen.csv", keep_default_na=False)
p = re.findall(r"[a-z]+", " ".join(d["texto"]).lower())
print("palavras no corpus:", len(p))
fr = sorted(Counter(p).items(), key=lambda kv: (-kv[1], kv[0]))
V = 2000; vocab = [w for w, _ in fr[:V]]; pos = {w: i for i, w in enumerate(vocab)}
ids = np.array([pos.get(w, -1) for w in p])
C = np.zeros((V, V))
for o in (1, 2):
    i, j = ids[:-o], ids[o:]
    ok = (i >= 0) & (j >= 0)
    np.add.at(C, (i[ok], j[ok]), 1)
C = C + C.T
N = C.sum(); pi = C.sum(1) / N
pc = C.sum(0) ** 0.75; pc /= pc.sum()
with np.errstate(divide="ignore"):
    M = np.log((C / N) / np.outer(pi, pc))
M[~np.isfinite(M) | (M < 0)] = 0
print(f"densidade = {(M > 0).mean():.4f}")
U, s, _ = np.linalg.svd(M)
print("valores singulares:", s[[0, 1, 9, 49, 99, 499]].round(2))
k = 100
print(f"fração de Frobenius em k = 100: {(s[:k] ** 2).sum() / (s ** 2).sum():.4f}")
W = U[:, :k] * np.sqrt(s[:k]); Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
for w in ("elizabeth", "happy", "letter", "mother", "walked", "london", "ten"):
    cs = Wn @ Wn[pos[w]]
    print(f"{w:10s}", ", ".join(vocab[i] for i in np.argsort(-cs)[1:7]))
