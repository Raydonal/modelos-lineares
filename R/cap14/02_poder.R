# Cap. 14: Poder do teste F e tamanho de amostra
source(file.path("R", "comum", "gerador.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
poder <- function(lam, q, nu, a = 0.05)
  pf(qf(1 - a, q, nu), q, nu, ncp = lam, lower.tail = FALSE)
# Dois grupos de r unidades, delta/sigma = 1: lambda = r/2
for (r in c(10, 15, 16, 17, 18, 20))
  pr("r %2d lambda %.1f poder %.4f", r, r / 2,
     poder(r / 2, 1, 2 * r - 2))
pr("aproximação normal: r = %.2f",
   2 * (qnorm(0.975) + qnorm(0.8))^2)
# Mesmo lambda, mais graus de liberdade no numerador: menos poder
for (q in c(1, 2, 3, 6, 12))
  pr("q %2d poder(lambda = 10, nu = 30) %.4f", q, poder(10, q, 30))
# Conferência por simulação: 3 grupos de 5, sigma = 2 (Cap. 10)
g <- gerador(1414); N <- 20000; rej <- 0
mu <- rep(c(10, 11, 13), each = 5); cr <- qf(0.95, 2, 12)
Z <- matriz_normal(g, N, 15, sd = 2)
for (i in seq_len(N)) {
  y <- mu + Z[i, ]
  mg <- c(mean(y[1:5]), mean(y[6:10]), mean(y[11:15]))
  qe <- sum((y - rep(mg, each = 5))^2) / 12
  qa <- 5 * sum((mg - mean(y))^2) / 2
  rej <- rej + (qa / qe > cr)
}
lam <- 5 * sum((c(10, 11, 13) - 34 / 3)^2) / 4
pr("lambda %.3f poder %.4f simulado %.4f", lam, poder(lam, 2, 12),
   rej / N)
