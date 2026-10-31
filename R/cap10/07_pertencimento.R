# Cap. 10: Ataques de pertencimento contra o mecanismo gaussiano
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
db <- read.csv(file.path("dados", "diabetes.csv"))
X <- as.matrix(db[, 1:10]); p <- ncol(X)
X <- sweep(sweep(X, 2, apply(X, 2, min)), 2,
           apply(X, 2, function(v) diff(range(v))), "/")
Delta <- sqrt(p); D <- sqrt(sum(X[1, ]^2))   # ||a - b|| = ||x_1||
sigma_suf <- function(eps, delta = 1e-5) {    # Cap. 9
  z <- qnorm(delta); u <- (sqrt(2) * z + sqrt(2 * z^2 + 4 * eps)) / 2
  Delta / (sqrt(2) * u)
}
alfa <- 0.05
t(sapply(c(0.5, 1, 2, 4, 8), function(eps) {
  s <- sigma_suf(eps)
  c(eps = eps, sigma = s,
    poder_NP = pnorm(qnorm(alfa) + D / s),    # conhece x_1
    NP_pior = pnorm(qnorm(alfa) + Delta / s), # registro extremo
    poder_qui2 = pchisq(qchisq(1 - alfa, p), p, ncp = (D / s)^2,
                        lower.tail = FALSE),  # não conhece x_1
    cota_DP = min(1, exp(eps) * alfa + 1e-5))
}))
# Simulação com eps = 4: 10 000 publicações com e sem o paciente 1
s <- sigma_suf(4); a <- colSums(X); b <- colSums(X[-1, ])
g <- gerador(43); R <- 10000
Y1 <- sweep(matriz_normal(g, R, p, s), 2, a, "+")   # presente
Y0 <- sweep(matriz_normal(g, R, p, s), 2, b, "+")   # ausente
np_est <- function(Y) drop(sweep(Y, 2, b) %*% (a - b)) / (s * D)
q2 <- function(Y) rowSums(sweep(Y, 2, b)^2) / s^2
c(alfa_NP = mean(np_est(Y0) > qnorm(1 - alfa)),
  poder_NP = mean(np_est(Y1) > qnorm(1 - alfa)),
  alfa_qui2 = mean(q2(Y0) > qchisq(1 - alfa, p)),
  poder_qui2 = mean(q2(Y1) > qchisq(1 - alfa, p)))
c(norma_x1 = D, Delta = Delta)
