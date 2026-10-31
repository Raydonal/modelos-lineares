"""Cap. 7: Quantização de embeddings: erro em normas e preservação de vizinhos."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "austen.csv", keep_default_na=False)
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


def quantiza(W, bits, por_linha):
    q = 2 ** (bits - 1) - 1
    esc = (np.abs(W).max(1, keepdims=True) if por_linha else np.abs(W).max()) / q
    return np.round(W / esc) * esc


def vizinhos(W, k=5):
    nr = np.linalg.norm(W, axis=1, keepdims=True); nr[nr == 0] = 1
    Wn = W / nr; S = Wn[:300] @ Wn.T; S[np.arange(300), np.arange(300)] = -np.inf
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


v0 = vizinhos(W)
for bits, pl in ((8, True), (8, False), (4, True), (4, False), (3, True), (2, True)):
    Q = quantiza(W, bits, pl); E = W - Q; vq = vizinhos(Q)
    ov = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(v0, vq)])
    print(f"{bits} bits, {'por linha' if pl else 'global   '}: erro espectral = {np.linalg.norm(E, 2) / np.linalg.norm(W, 2):.4f}, "
          f"Frobenius = {np.linalg.norm(E) / np.linalg.norm(W):.4f}, vizinhos preservados = {ov:.4f}")
