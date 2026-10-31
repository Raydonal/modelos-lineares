# Cap. 10: Qui-quadrado não central, mistura de Poisson e a F
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
g <- gerador(29); N <- 20000; k <- 3
mu <- c(2, 0, 0)                              # lambda = ||mu||^2 = 4
Z <- matriz_normal(g, N, k)
u <- rowSums(sweep(Z, 2, mu, "+")^2)          # ||mu + z||^2
q95 <- qchisq(0.95, k)
mistura <- sum(dpois(0:200, 2) * pchisq(q95, k + 2 * (0:200),
                                        lower.tail = FALSE))
c(media = mean(u), teorica = k + 4, variancia = var(u),
  teorica = 2 * k + 4 * 4, P_simulada = mean(u > q95),
  P_pchisq = pchisq(q95, k, ncp = 4, lower.tail = FALSE),
  P_mistura = mistura)
# Poder do teste qui-quadrado: cresce com lambda
lam <- c(0, 1, 4, 9, 16)
round(rbind(lambda = lam,
  poder = pchisq(q95, k, ncp = lam, lower.tail = FALSE)), 4)
# t^2 com nu gl é F(1, nu)
c(t2 = qt(0.975, 10)^2, F = qf(0.95, 1, 10))
