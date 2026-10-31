"""Cap. 1 -- Camadas lineares: composição de funções afins é afim."""
import numpy as np

rng = np.random.default_rng(1)
W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)   # camada 1: R^3 -> R^4
W2, b2 = rng.normal(size=(2, 4)), rng.normal(size=2)   # camada 2: R^4 -> R^2
x = np.array([1.0, -2.0, 0.5])

rede = lambda x: W2 @ (W1 @ x + b1) + b2               # duas camadas, sem ativação
W, b = W2 @ W1, W2 @ b1 + b2                           # uma camada equivalente
print(rede(x), W @ x + b)


def aumentar(W, b):
    """Matriz (m+1) x (n+1) que representa x -> Wx + b em coordenadas homogêneas."""
    m, n = W.shape
    return np.block([[W, b[:, None]], [np.zeros((1, n)), np.ones((1, 1))]])


M = aumentar(W2, b2) @ aumentar(W1, b1)
print(M @ np.append(x, 1))                             # última coordenada = 1

relu = lambda z: np.maximum(z, 0)
rede_relu = lambda x: W2 @ relu(W1 @ x + b1) + b2
print(rede_relu(2 * x), 2 * rede_relu(x) - b2)         # diferem em geral
