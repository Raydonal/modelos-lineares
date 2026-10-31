# Cap. 14: Sobreidentificação: a estatística de Sargan
source(file.path("R", "comum", "gerador.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
proj <- function(Z, v) Z %*% solve(crossprod(Z), crossprod(Z, v))
sargan <- function(y, X, Z) {
  Xh <- proj(Z, X)
  b <- solve(crossprod(Xh, X), crossprod(Xh, y))
  e <- y - X %*% b
  list(b = drop(b), e = drop(e), num = sum(proj(Z, e)^2),
       S = sum(proj(Z, e)^2) / (sum(e^2) / length(y)))
}
# Exemplo à mão: Z ortonormal, Z'x = (2, 1), Z'y = (1, 1)
Z <- cbind(rep(1, 4), c(1, 1, -1, -1)) / 2
x <- c(1.5, 1.5, 0.5, 0.5); y <- c(1.5, 0.5, 0.5, -0.5)
s <- sargan(y, cbind(x), Z)
pr("Z'x %.1f %.1f  Z'y %.1f %.1f", sum(Z[, 1] * x), sum(Z[, 2] * x),
   sum(Z[, 1] * y), sum(Z[, 2] * y))
pr("b %.2f  e %s", s$b, paste(sprintf("%.1f", s$e), collapse = " "))
pr("e'PZe %.2f  e'e/n %.2f  S %.4f p %.3f", s$num, sum(s$e^2) / 4,
   s$S, pchisq(s$S, 1, lower.tail = FALSE))
# Identidade: e'PZe = eps'(PZ - P_Xh)eps
g <- gerador(77); n <- 12
Z <- cbind(1, matriz_normal(g, n, 3)); X <- cbind(1, g$normal(n))
eps <- g$normal(n); beta <- c(1, 2); y <- drop(X %*% beta) + eps
s <- sargan(y, X, Z); Xh <- proj(Z, X)
pr("identidade: %.10f %.10f", s$num,
   sum(proj(Z, eps)^2) - sum(proj(Xh, eps)^2))
# Simulação: n = 100, k = 3, p = 2 (um grau de sobreidentificação)
simula <- function(c1, c2, N = 4000, n = 100) {
  g <- gerador(1958); rej <- 0; bb <- 0; cr <- qchisq(0.95, 1)
  for (i in seq_len(N)) {
    z1 <- g$normal(n); z2 <- g$normal(n)
    v <- g$normal(n); e <- g$normal(n)
    eps <- 0.7 * v + sqrt(0.51) * e
    x <- 0.5 * z1 + 0.5 * z2 + v
    y <- x + c1 * z1 + c2 * z2 + eps
    s <- sargan(y, cbind(1, x), cbind(1, z1, z2))
    rej <- rej + (s$S > cr); bb <- bb + s$b[2]
  }
  c(rej / N, bb / N)
}
for (cc in list(c(0, 0), c(0, 0.3), c(0.3, 0.3))) {
  v <- simula(cc[1], cc[2])
  pr("efeito direto (%.1f, %.1f): rejeita %.3f  média beta %.3f",
     cc[1], cc[2], v[1], v[2])
}
