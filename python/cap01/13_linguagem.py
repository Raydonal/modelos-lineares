"""Cap. 1: Modelos de linguagem pela álgebra linear: tokens (BPE), embeddings e busca por similaridade."""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "austen.csv", keep_default_na=False)
palavras = lambda txt: re.findall(r"[a-z]+", txt.lower())

# (1) BPE treinado em Orgulho e Preconceito
pp = palavras(" ".join(d.loc[d["livro"] == "Pride & Prejudice", "texto"]))
freq = {w: c for w, c in sorted(Counter(pp).items()) if c >= 2}
w = {p: " " + " ".join(p) + " _ " for p in freq}
fundir = lambda x, a, b: re.sub(f"(?<= ){a} {b}(?= )", a + b, x)
regras = []
for _ in range(200):
    cont = Counter()
    for p, x in w.items():
        s = x.split()
        for a, b in zip(s[:-1], s[1:]):
            cont[(a, b)] += freq[p]
    a, b = min(cont, key=lambda k: (-cont[k], k))           # empates: ordem lexicográfica
    regras.append((a, b))
    w = {p: fundir(x, a, b) for p, x in w.items()}
print("primeiras regras:", [" ".join(r) for r in regras[:12]])
vocab = sorted(set(s for x in w.values() for s in x.split()) | set("abcdefghijklmnopqrstuvwxyz_"))
print("tamanho do vocabulário:", len(vocab))


def tokenizar(p):
    x = " " + " ".join(p) + " _ "
    for a, b in regras:
        x = fundir(x, a, b)
    return x.split()


frase = "the unbelievably proud gentleman danced reluctantly"
toks = [t for p in palavras(frase) for t in tokenizar(p)]
print(toks, f"\n{len(palavras(frase))} palavras, {len(toks)} tokens")

# (2) Embeddings: consulta à tabela = produto pela matriz one-hot
E = Gerador(1).matriz_normal(len(vocab), 8)
ids = [vocab.index(t) for t in toks]
O = np.zeros((len(ids), len(vocab))); O[np.arange(len(ids)), ids] = 1
print("max|OE - E[ids]| =", np.abs(O @ E - E[ids]).max())

# (3) Busca por similaridade entre capítulos (TF-IDF e cosseno)
cap = d.groupby("livro", sort=False)["texto"].transform(
    lambda t: t.str.contains(r"^chapter [0-9ivxlc]+", case=False, regex=True).cumsum())
d = d.assign(doc=d["livro"] + " - cap. " + cap.astype(str))[cap > 0]
txt = d.groupby("doc", sort=False)["texto"].apply(" ".join)
tok = [palavras(t) for t in txt]
termos = sorted(set(t for ws in tok for t in ws)); pos = {t: i for i, t in enumerate(termos)}
tf = np.zeros((len(tok), len(termos)))
for i, ws in enumerate(tok):
    np.add.at(tf[i], [pos[t] for t in ws], 1)
idf = np.log(len(tok) / (tf > 0).sum(0))
X = tf * idf; X /= np.linalg.norm(X, axis=1, keepdims=True)
q = np.zeros(len(termos))
for t in palavras("a ball at netherfield where elizabeth dances with darcy"):
    if t in pos:
        q[pos[t]] += 1
q = q * idf; q /= np.linalg.norm(q)
cos = X @ q
for i in np.argsort(-cos)[:4]:
    print(f"{txt.index[i]:30s} {cos[i]:.3f}")
