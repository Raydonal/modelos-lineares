# Cap. 4 -- Interpolação de norma mínima e dupla descida (modelo de "features fracas")
set.seed(2026)
D <- 500; n <- 50; sigma <- 0.5
beta <- rep(2 / sqrt(D), D)                      # 500 efeitos fracos, ||beta|| = 2
ps <- c(5, 10, 20, 30, 40, 45, 48, 50, 52, 55, 60, 80, 100, 200, 300, 500)
risco <- function(p, X, y) {                     # usa só as p primeiras covariáveis
  b <- MASS::ginv(X[, 1:p, drop = FALSE]) %*% y  # mínimos quadrados de norma mínima
  sum((beta[1:p] - b)^2) + sum(beta[-(1:p)]^2) + sigma^2   # E(y0 - x0'b)^2, x0 ~ N(0, I)
}
R <- replicate(200, {
  X <- matrix(rnorm(n * D), n); y <- X %*% beta + sigma * rnorm(n)
  sapply(ps, function(p) risco(p, X, y))
})
setNames(round(apply(R, 1, median), 2), ps)    # risco mediano em função de p
c(risco_preditor_nulo = sum(beta^2) + sigma^2)

# Com p > n, a solução de norma mínima interpola e coincide com X'(XX')^{-1} y
X <- matrix(rnorm(n * 200), n); y <- rnorm(n)
b1 <- MASS::ginv(X) %*% y; b2 <- t(X) %*% solve(tcrossprod(X), y)
c(dif = max(abs(b1 - b2)), residuo = max(abs(X %*% b1 - y)))
