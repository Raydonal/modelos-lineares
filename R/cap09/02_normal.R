# Cap. 9: Normal multivariada: densidade, condicionais, singular
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
mu <- c(1, 2, 0)
Sigma <- matrix(c(4, 2, 1, 2, 3, -1, 1, -1, 2), 3)
L <- t(chol(Sigma))

# Log-densidade pela Cholesky: z = L^{-1}(x - mu),
# log det(Sigma) = 2 sum log L_ii
logdens <- function(x) {
  z <- forwardsolve(L, x - mu)
  -0.5 * (3 * log(2 * pi) + sum(z^2)) - sum(log(diag(L)))
}
x0 <- c(2, 1, 1)
c(cholesky = logdens(x0),
  formula = -0.5 * (3 * log(2 * pi) + log(det(Sigma)) +
    drop(t(x0 - mu) %*% solve(Sigma, x0 - mu))))

# Condicional de x1 dado (x2, x3) = (3, -1): complemento de Schur
i1 <- 1; i2 <- 2:3; x2 <- c(3, -1)
B <- Sigma[i1, i2] %*% solve(Sigma[i2, i2])
c(media = mu[i1] + drop(B %*% (x2 - mu[i2])),
  variancia = Sigma[i1, i1] - drop(B %*% Sigma[i2, i1]))

# Simulação: 100 000 vetores; regressão de x1 em (x2, x3)
g <- gerador(11); N <- 1e5
X <- sweep(matriz_normal(g, N, 3) %*% t(L), 2, mu, "+")
round(c(colMeans(X), cov(X)[upper.tri(Sigma, TRUE)]), 3)
ajuste <- lm(X[, 1] ~ X[, 2] + X[, 3])
round(c(coef(ajuste), var_resid = var(resid(ajuste))), 3)
round(c(B = B), 3)

# Distância de Mahalanobis: d2 = z'z ~ qui-quadrado com 3 gl
d2 <- colSums(forwardsolve(L, t(X) - mu)^2)
c(media = mean(d2), quantil95 = unname(quantile(d2, 0.95)),
  qchisq95 = qchisq(0.95, 3))

# Normal singular: x = (z1, z2, z1 + z2), no plano x1 + x2 = x3
S2 <- matrix(c(1, 0, 1, 0, 1, 1, 1, 1, 2), 3)
e <- eigen(S2); round(e$values, 12)
round(e$vectors[, 3], 4)               # base de N(Sigma)

# Não correlacionadas mas dependentes: Y = S X, S = +-1 ao acaso
xs <- g$normal(N); s <- ifelse(g$unif(N) < 0.5, -1, 1); ys <- s * xs
c(correlacao = cor(xs, ys), cor_quadrados = cor(xs^2, ys^2))
