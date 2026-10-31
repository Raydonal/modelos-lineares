# Cap. 4 -- O gradiente descendente converge para a solução de norma mínima
set.seed(7)
n <- 20; p <- 100
X <- matrix(rnorm(n * p), n); y <- rnorm(n)
Xp <- MASS::ginv(X); b_mn <- drop(Xp %*% y)               # solução de norma mínima
eta <- 1 / max(svd(X)$d)^2                                 # passo (Capítulo 7)
gd <- function(b, iter = 3000) {
  for (k in 1:iter) b <- b - eta * drop(crossprod(X, X %*% b - y))
  b
}
b_zero <- gd(rep(0, p))                                    # início na origem
b0 <- rnorm(p); b_alea <- gd(b0)                           # início aleatório
Pn <- diag(p) - Xp %*% X                                   # projeção sobre N(X)
c(residuo_zero = max(abs(X %*% b_zero - y)),
  dist_zero_mn = sqrt(sum((b_zero - b_mn)^2)),
  dist_alea_mn = sqrt(sum((b_alea - b_mn)^2)),
  dist_alea_previsto = sqrt(sum((b_alea - b_mn - Pn %*% b0)^2)))
c(norma_mn = sqrt(sum(b_mn^2)), norma_alea = sqrt(sum(b_alea^2)))
