# Cap. 10 -- Wishart, complemento de Schur e o T^2 de Hotelling
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
p <- 3; n <- 12; R <- 20000
mu <- c(1, 2, 0)
Sigma <- matrix(c(4, 2, 1, 2, 3, -1, 1, -1, 2), 3)
L <- t(chol(Sigma)); g <- gerador(37)
T2 <- schur <- numeric(R)
for (r in 1:R) {
  X <- sweep(matriz_normal(g, n, p) %*% t(L), 2, mu, "+")
  xb <- colMeans(X); A <- crossprod(sweep(X, 2, xb))
  S <- A / (n - 1)
  T2[r] <- n * drop(t(xb - mu) %*% solve(S, xb - mu))
  W <- tcrossprod(forwardsolve(L, t(sweep(X, 2, xb))))  # ~ W(n-1, I)
  schur[r] <- 1 / solve(W)[1, 1]              # = W11 - W12 W22^-1 W21
}
# 1 / (W^{-1})_11 ~ qui-quadrado com m - p + 1 = 9 gl (aqui m = n - 1)
c(media_schur = mean(schur), teorica = n - 1 - p + 1)
Fst <- (n - p) / (p * (n - 1)) * T2
round(rbind(quantis = quantile(Fst, c(0.5, 0.9, 0.95, 0.99)),
            F = qf(c(0.5, 0.9, 0.95, 0.99), p, n - p)), 4)
c(rejeicao_5 = mean(Fst > qf(0.95, p, n - p)),
  rejeicao_qui2 = mean(T2 > qchisq(0.95, p)))
