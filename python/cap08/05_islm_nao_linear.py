"""Cap. 8 -- IS-LM não linear: estática comparativa pelo teorema da função implícita."""
import numpy as np

c0, c1, T = 100, 0.75, 200
F = lambda x, G, M: np.array([x[0] - c0 - c1 * (x[0] - T) - 800 * (x[1] / 5) ** -0.3125 - G,
                              M - 500 * (x[0] / 4000) ** 2 * (x[1] / 5) ** -1])
Jx = lambda x: np.array([[1 - c1, 0.3125 * 800 / 5 * (x[1] / 5) ** -1.3125],
                         [-1000 / 4000 * (x[0] / 4000) * (x[1] / 5) ** -1, 500 * (x[0] / 4000) ** 2 * (x[1] / 5) ** -2 / 5]])
Jt = np.array([[-1.0, 0], [0, 1]])


def newton(G, M, x=(4000.0, 5.0)):
    z = np.array([x[0], np.log(x[1])])
    for _ in range(100):
        x = np.array([z[0], np.exp(z[1])]); z = z - np.linalg.solve(Jx(x) @ np.diag([1, x[1]]), F(x, G, M))
    return np.array([z[0], np.exp(z[1])])


x0 = newton(250, 500)
print("equilíbrio:", x0.round(4), "\nmultiplicadores (TFI):\n", (-np.linalg.solve(Jx(x0), Jt)).round(5))
for M in (500, 750, 1000, 1500, 2000, 3000):
    x = newton(250, M); m = -np.linalg.solve(Jx(x), Jt)
    print(f"M/P = {M:4d}: Y = {x[0]:8.3f}, r = {x[1]:6.3f}, dY/dM = {m[0, 1]:.3f}")
