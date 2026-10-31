"""Embeddings de palavras dos romances de Jane Austen pela SVD da matriz PPMI.

A construção é explicada no Cap. 6 (python/cap06/11_embeddings.py). Devolve o
vocabulário, as coocorrências C e os embeddings normalizados E (k colunas).
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def embeddings_austen(V=2000, k=100):
    raiz = Path(__file__).resolve().parents[2]
    d = pd.read_csv(raiz / "dados" / "austen.csv", keep_default_na=False)
    p = re.findall(r"[a-z]+", " ".join(d["texto"]).lower())
    fr = sorted(Counter(p).items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [w for w, _ in fr[:V]]
    pos = {w: i for i, w in enumerate(vocab)}
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
    U, s, _ = np.linalg.svd(M)
    W = U[:, :k] * np.sqrt(s[:k])
    return vocab, C, W / np.linalg.norm(W, axis=1, keepdims=True)
