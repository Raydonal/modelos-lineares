"""Cap. 5: Sanitização de embeddings: projetar fora o gênero."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.austen import embeddings_austen  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

vocab, C, E = embeddings_austen()
ind = {w: i for i, w in enumerate(vocab)}
pos = lambda ws: np.array([ind[w] for w in ws])

# 1. Direção de gênero: média das diferenças em pares definidores
pares = [("he", "she"), ("his", "her"), ("himself", "herself"),
         ("mr", "mrs"), ("sir", "lady"), ("man", "woman"),
         ("men", "women"), ("gentleman", "lady"),
         ("gentlemen", "ladies"), ("brother", "sister"),
         ("brothers", "sisters"), ("father", "mother"),
         ("son", "daughter"), ("husband", "wife"),
         ("uncle", "aunt"), ("nephew", "niece"), ("boy", "girl")]
m, f = zip(*pares)
v = (E[pos(f)] - E[pos(m)]).mean(0); v /= np.linalg.norm(v)
nomes = """darcy edmund knightley thomas john henry edward william
frank wickham charles willoughby brandon walter james tom robert
frederick bingley collins wentworth tilney rushworth benwick
fanny emma elizabeth elinor catherine jane marianne anne harriet
mary isabella lucy lydia maria julia louisa charlotte lizzy
eleanor kitty susan henrietta margaret eliza""".split()
y_n = np.repeat([-1.0, 1.0], 24)       # -1 masculino, +1 feminino
acerto = np.mean(np.sign(E[pos(nomes)] @ v) == y_n)
print("acerto dos nomes por v:", acerto)

# 2. Rótulos: palavras que coocorrem mais com she/her ou he/his
cf = C[:, ind["she"]] + C[:, ind["her"]]
cm = C[:, ind["he"]] + C[:, ind["his"]]
r = np.log((cf + 1) / (cm + 1)) - np.log(cf.sum() / cm.sum())
fora = set(pos(m + f + tuple(nomes)))
o = np.array([i for i in np.argsort(r, kind="stable")
              if i not in fora])
K = 300; sel = np.r_[o[:K], o[::-1][:K]]
y = np.repeat([-1.0, 1.0], K)
print("mais masculinas:", " ".join(vocab[i] for i in o[:8]))
print("mais femininas: ", " ".join(vocab[i] for i in o[::-1][:8]))
perm = np.argsort(Gerador(7).unif(2 * K), kind="stable")
tr, y_tr = sel[perm[:420]], y[perm[:420]]
te, y_te = sel[perm[420:]], y[perm[420:]]


# 3. INLP: classificador de MQ em C(Q)^perp; remove sua direção
def complemento(Q):
    if Q.shape[1] == 0:
        return np.eye(E.shape[1])
    return np.linalg.qr(Q, mode="complete")[0][:, Q.shape[1]:]


def knn(Z, k=5):                       # k vizinhos entre os nomes
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    viz = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.mean(np.sign(y_n[viz].sum(1)) == y_n)


def avaliar(Q):
    B = complemento(Q); Z = E[tr] @ B; mu = Z.mean(0)
    gam = np.linalg.lstsq(Z - mu, y_tr, rcond=None)[0]
    prev = lambda i: np.sign((E[i] @ B - mu) @ gam)
    acc = (np.mean(prev(te) == y_te),
           np.mean(prev(pos(nomes)) == y_n),
           knn(E[pos(nomes)] @ B))
    return B @ gam / np.linalg.norm(gam), acc


Q = np.zeros((E.shape[1], 0))
print("  r  palavras  nomes  nomes_knn")
for r_ in range(41):
    w, acc = avaliar(Q)
    if r_ in (0, 1, 2, 3, 5, 10, 20, 40):
        print(f"{r_:3d}" + "".join(f"{a:9.3f}" for a in acc))
    if r_ == 10:
        Q10 = Q                        # as 10 primeiras direções
    Q = np.column_stack([Q, v if r_ == 0 else w])  # 1a: v; INLP
print("ortonormalidade:", np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())

# 4. O que a projeção P = I - QQ' (r = 10) preserva
P = np.eye(E.shape[1]) - Q10 @ Q10.T
cs = np.corrcoef((E @ E.T).ravel(), (E @ P @ E.T).ravel())[0, 1]
print(f"norma removida = {np.mean(np.sum((E @ Q10)**2, 1)):.4f},"
      f" correlação dos cossenos = {cs:.4f}")
Es = E @ P; Es /= np.linalg.norm(Es, axis=1, keepdims=True)


def viz(M, w):
    o = np.argsort(-(M @ M[ind[w]]), kind="stable")[1:6]
    return ", ".join(vocab[i] for i in o)


for w in ("happy", "letter", "london", "mother", "elizabeth"):
    print(f"{w:9s} antes: {viz(E, w)}\n{'':9s} depois: {viz(Es, w)}")
