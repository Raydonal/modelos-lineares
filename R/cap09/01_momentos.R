# Cap. 9 -- Momentos: Var(Ax), viés de A/n e E(x'Ax)
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
mu <- c(1, 2, 0)
Sigma <- matrix(c(4, 2, 1, 2, 3, -1, 1, -1, 2), 3)
L <- t(chol(Sigma))                    # Sigma = L L'
A <- matrix(c(1, 1, 0, 0, 1, -1), 2, byrow = TRUE)
A %*% Sigma %*% t(A)                   # Var(Ax) = A Sigma A'

# 20 000 amostras de tamanho n = 5 (normais, via x = mu + L z)
g <- gerador(9); R <- 20000; n <- 5
soma_S <- matrix(0, 3, 3); soma_q <- 0; B <- diag(c(1, 2, 3))
for (r in 1:R) {
  Z <- matriz_normal(g, n, 3)          # linhas: z_i'
  X <- sweep(Z %*% t(L), 2, mu, "+")   # linhas: x_i' = (mu + L z_i)'
  A_r <- crossprod(sweep(X, 2, colMeans(X)))
  soma_S <- soma_S + A_r / n           # estimador de MV: A / n
  soma_q <- soma_q + sum((X %*% B) * X) / n   # média de x_i' B x_i
}
round(soma_S / R, 3)                   # perto de (n - 1)/n Sigma
(n - 1) / n * Sigma
c(simulado = soma_q / R,
  teorico = sum(diag(B %*% Sigma)) + drop(t(mu) %*% B %*% mu))
