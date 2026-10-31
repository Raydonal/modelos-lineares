"""Cap. 1 -- Conhecendo os romances de Jane Austen"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "austen.csv", keep_default_na=False)


def pal(t):
    return re.findall(r"[a-z]+", " ".join(t).lower())


for livro, t in d.groupby("livro")["texto"]:
    p = pal(t)
    print(f"{livro:<22}{len(p):8d}{len(set(p)):7d}")
fr = Counter(pal(d["texto"]))
f = sorted(fr.values(), reverse=True)
print("total", sum(f), " distintas", len(f),
      " uma vez", sum(v == 1 for v in f))
top = [f"{w}:{c}" for w, c in fr.most_common(10)]
print(" ".join(top[:5])); print(" ".join(top[5:]))
# Lei de Zipf: frequência x posto é quase constante
for k in [1, 10, 100, 1000, 10000]:
    print(f"posto {k:6d}  freq {f[k - 1]:6d}  freq x posto "
          f"{f[k - 1] * k:7d}")
