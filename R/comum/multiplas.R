# Comparações múltiplas: valor crítico de Dunnett e Benjamini-Hochberg
# (mesmas contas de python/comum/multiplas.py).
# P(|T_i| <= d, i = 1..k), k tratamentos e um controle, n iguais:
# E[ (Phi(z + sqrt2 d s) - Phi(z - sqrt2 d s))^k ], em z e em s.
conf_dunnett <- function(d, k, nu, M = 400) {
  z <- seq(-8, 8, by = 0.05); w <- dnorm(z) * 0.05
  s <- sqrt(qchisq((seq_len(M) - 0.5) / M, nu) / nu)
  mean(sapply(s, function(si) {
    c0 <- sqrt(2) * d * si
    sum(w * (pnorm(z + c0) - pnorm(z - c0))^k)
  }))
}
q_dunnett <- function(k, nu, alfa = 0.05) {
  a <- 1; b <- 12
  for (i in 1:30) {
    d <- (a + b) / 2
    if (conf_dunnett(d, k, nu) < 1 - alfa) a <- d else b <- d
  }
  (a + b) / 2
}
# Número de rejeições de Benjamini-Hochberg
bh <- function(p, alfa = 0.05) {
  m <- length(p); o <- sort(p)
  k <- which(o <= alfa * seq_len(m) / m)
  if (length(k)) max(k) else 0L
}
