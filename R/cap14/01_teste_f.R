# Cap. 14: Teste F da hipótese linear geral: três contas iguais
pr <- function(...) cat(sprintf(...), "\n", sep = "")
# Dois grupos, modelo superparametrizado (posto 2 em 3 colunas)
y <- c(1, 3, 4, 6, 5); n <- 5
X <- cbind(1, c(1, 1, 0, 0, 0), c(0, 0, 1, 1, 1))
K <- rbind(c(0, 1, -1)); m <- 0; q <- 1
r <- qr(X)$rank
Gs <- list(diag(c(0, 1 / 2, 1 / 3)), MASS::ginv(crossprod(X)))
for (G in Gs) {
  b <- drop(G %*% crossprod(X, y))
  d <- drop(K %*% b) - m
  KGK <- drop(K %*% G %*% t(K))
  Q <- d^2 / KGK
  s2 <- sum((y - X %*% b)^2) / (n - r)
  pr("Kb %.4f KGK' %.4f Q %.4f s2 %.4f F %.4f", d, KGK, Q, s2,
     Q / q / s2)
}
# Modelos encaixados: SQR restrita - SQR irrestrita
G <- Gs[[1]]; b <- drop(G %*% crossprod(X, y))
KGK <- K %*% G %*% t(K)
bK <- b - drop(G %*% t(K) %*% solve(KGK, K %*% b - m))
sqr0 <- sum((y - X %*% bK)^2); sqr1 <- sum((y - X %*% b)^2)
Fobs <- (sqr0 - sqr1) / q / (sqr1 / (n - r))
pr("SQR0 %.4f SQR1 %.4f F %.4f", sqr0, sqr1, Fobs)
pr("p %.4f crit5%% %.3f", pf(Fobs, q, n - r, lower.tail = FALSE),
   qf(0.95, q, n - r))
# Geometria: P1 - P0 é a projeção em Col(XGK')
A <- X %*% G %*% t(K)
PA <- A %*% solve(crossprod(A), t(A))
P1 <- X %*% G %*% t(X)
P0 <- matrix(1 / n, n, n)
pr("max|P1-P0-PA| %.1e  y'PAy %.4f", max(abs(P1 - P0 - PA)),
   sum(y * (PA %*% y)))
# Hipótese não estimável: alfa_1 = 0 não muda o ajuste
K2 <- rbind(c(0, 1, 0))
Z <- X[, -2]
pr("SQR sob alfa1=0 %.4f (igual a SQR1 %.4f)",
   sum((y - Z %*% solve(crossprod(Z), crossprod(Z, y)))^2), sqr1)
# Três pontos: H0 beta_1 = 0 e H0 (beta_0, beta_1) = (1, 1)
x <- 0:2; y3 <- c(1, 2, 4); X3 <- cbind(1, x)
A3 <- solve(crossprod(X3))
b3 <- drop(A3 %*% crossprod(X3, y3))
s23 <- sum((y3 - X3 %*% b3)^2) / 1
d <- b3 - c(1, 1)
F2 <- drop(t(d) %*% solve(A3, d)) / 2 / s23
pr("3 pontos: b (%.4f, %.4f) s2 %.4f F(q=1) %.1f F(q=2) %.2f",
   b3[1], b3[2], s23, b3[2]^2 / A3[2, 2] / s23, F2)
