"""Cap. 1 -- Índices de Laspeyres, Paasche e Fisher como razões de produtos internos."""
import numpy as np
import pandas as pd

cesta = pd.DataFrame({
    "bem": ["arroz", "feijão", "carne", "leite", "pão"],
    "p0": [5.20, 8.90, 38.50, 4.60, 12.00],   # preços no período base (R$/unidade)
    "p1": [6.10, 8.10, 44.90, 5.30, 13.20],   # preços no período atual
    "q0": [10, 4, 6, 20, 8],                  # quantidades no período base
    "q1": [11, 5, 4.5, 19, 8],                # quantidades no período atual
})
p0, p1, q0, q1 = (cesta[c].to_numpy(float) for c in ("p0", "p1", "q0", "q1"))

laspeyres = (p1 @ q0) / (p0 @ q0)
paasche = (p1 @ q1) / (p0 @ q1)
fisher = np.sqrt(laspeyres * paasche)
print(f"Laspeyres = {laspeyres:.4f}, Paasche = {paasche:.4f}, Fisher = {fisher:.4f}")

w = p0 * q0 / (p0 @ q0)                       # pesos de dispêndio no período base
print(pd.DataFrame({"bem": cesta["bem"], "peso": w.round(3), "relativo": (p1 / p0).round(3)}))
print(w @ (p1 / p0))                          # = Laspeyres
