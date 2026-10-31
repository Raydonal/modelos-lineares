"""Cap. 6 -- Velocidade de convergência do mercado de trabalho para o equilíbrio."""
import numpy as np

P = np.array([[0.960, 0.015, 0.025], [0.250, 0.600, 0.150], [0.040, 0.030, 0.930]])
val, S = np.linalg.eig(P)
ordem = np.argsort(-val.real); val, S = val[ordem].real, S[:, ordem]
pi = np.linalg.inv(S)[0].real; pi = pi / pi.sum()
print("autovalores:", val.round(7), " pi:", pi.round(4))
print(f"meia-vida = {np.log(2) / -np.log(val[1]):.4f} meses")
x = np.array([0.0, 1, 0])
for t in (1, 3, 6, 12, 24, 36):
    d = np.abs(x @ np.linalg.matrix_power(P, t) - pi).sum()
    print(f"t = {t:2d}: distância = {d:.4f}, distância / lambda2^t = {d / val[1] ** t:.4f}")
