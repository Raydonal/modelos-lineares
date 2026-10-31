# Cap. 1 -- Uma rede linear de duas camadas treinada por gradiente aprende
# a mesma função que a regressão de mínimos quadrados
set.seed(42)
n <- 500; p <- 5; h <- 8
X <- matrix(rnorm(n * p), n, p)
beta <- c(2, -1, 0.5, 0, 3)
y <- X %*% beta + rnorm(n, sd = 0.5)

W1 <- matrix(rnorm(h * p, sd = 0.3), h, p)       # camada 1: R^5 -> R^8
w2 <- matrix(rnorm(h, sd = 0.3), 1, h)           # camada 2: R^8 -> R
eta <- 0.02
for (it in 1:3000) {
  r <- X %*% t(W1) %*% t(w2) - y                 # resíduos (n x 1)
  g_prod <- t(X) %*% r / n                       # gradiente em relação a W = w2 W1
  g_w2 <- t(g_prod) %*% t(W1)                    # regra da cadeia
  g_W1 <- t(w2) %*% t(g_prod)
  w2 <- w2 - eta * g_w2
  W1 <- W1 - eta * g_W1
}
rbind(rede = as.vector(w2 %*% W1),               # produto das camadas
      mqo  = coef(lm(y ~ X - 1)))
