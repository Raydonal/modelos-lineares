# Cap. 5 -- A atenção como suavizador: temperatura, projeções e colapso de posto
softmax <- function(M) { M <- M - apply(M, 1, max); E <- exp(M); E / rowSums(E) }
posto <- function(M) qr(M, tol = 1e-8)$rank
source(file.path("R", "comum", "gerador.R"))           # mesmos sorteios que o Python
g <- gerador(2026)
d <- 8; grupo <- rep(1:3, c(5, 4, 3)); n <- length(grupo)
centros <- matriz_normal(g, 3, d, sd = 1.5)
K <- centros[grupo, ]                                   # chaves: iguais dentro do grupo
Q <- K + matriz_normal(g, n, d, sd = 0.3)              # consultas: próximas das chaves
G <- model.matrix(~ factor(grupo) - 1)
PG <- G %*% solve(crossprod(G), t(G))                   # média dentro de cada grupo
atencao <- function(tau) softmax(Q %*% t(K) / (tau * sqrt(d)))
t(sapply(c(10, 3, 1, 0.3, 0.1), function(tau) {
  A <- atencao(tau)
  c(tau = tau, dist_PG = max(abs(A - PG)), erro_idemp = max(abs(A %*% A - A)),
    assimetria = max(abs(A - t(A))), traco = sum(diag(A)), posto = posto(A))
}))

# Camadas empilhadas sem conexões residuais: A^k -> 1 pi' (colapso de posto)
X <- matriz_normal(g, n, d)                             # tokens genéricos
A <- softmax(X %*% t(X) / sqrt(d))
ev <- eigen(t(A)); pi <- Re(ev$vectors[, 1]); pi <- pi / sum(pi)
L <- matrix(1, n) %*% t(pi)
c(erro_idemp_L = max(abs(L %*% L - L)), erro_AL = max(abs(A %*% L - L)), posto_L = posto(L))
Ak <- diag(n); dist <- numeric(64)
for (k in 1:64) { Ak <- Ak %*% A; dist[k] <- max(abs(Ak - L)) }
round(dist[c(1, 2, 4, 8, 16, 32, 64)], 6)
round(Mod(ev$values)[1:3], 4)                            # |lambda_2| governa a velocidade (Cap. 6)
# Com conexão residual, (I + A)/2 também é estocástica e tem o mesmo limite
Rk <- diag(n); for (k in 1:64) Rk <- Rk %*% ((diag(n) + A) / 2)
max(abs(Rk - L))

# Normalização de camada: projeção sobre 1-perp seguida de reescala para a esfera de raio sqrt(d)
ln <- function(x) { z <- x - mean(x); z / sqrt(mean(z^2)) }
x <- drop(matriz_normal(g, 1, d))
Cd <- diag(d) - matrix(1 / d, d, d)                         # matriz de centragem (projeção)
c(soma = sum(ln(x)), norma = sqrt(sum(ln(x)^2)), raiz_d = sqrt(d),
  invariancia = max(abs(ln(3 * x + 5) - ln(x))), via_projecao = max(abs(ln(x) - Cd %*% x / sqrt(mean((Cd %*% x)^2)))))
