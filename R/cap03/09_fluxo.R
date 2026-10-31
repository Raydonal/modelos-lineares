# Cap. 3: Fluxo normalizante com uma camada de acoplamento afim (Jacobiano triangular)
# Gerador: z ~ N(0, I); x1 = z1; x2 = t(x1) + exp(s(x1)) z2
s_fun <- function(x1) log(0.5) + 0.3 * x1
t_fun <- function(x1) 0.5 * x1^2 - 1
inversa <- function(x) cbind(x[, 1], (x[, 2] - t_fun(x[, 1])) / exp(s_fun(x[, 1])))
log_dens <- function(x) {                       # mudança de variáveis
  z <- inversa(x)
  rowSums(dnorm(z, log = TRUE)) - s_fun(x[, 1]) # log|det J_inversa| = -s(x1)
}
# Conferência do Jacobiano por diferenças finitas num ponto
x0 <- c(0.7, 0.4); h <- 1e-6
J <- sapply(1:2, function(j) { d <- c(0, 0); d[j] <- h
  (inversa(rbind(x0 + d)) - inversa(rbind(x0 - d))) / (2 * h) })
c(log_det_numerico = log(abs(det(J))), log_det_formula = -s_fun(x0[1]))
# A densidade integra 1 (grade)
g1 <- seq(-6, 6, length.out = 401); g2 <- seq(-4, 20, length.out = 801)
G <- as.matrix(expand.grid(g1, g2))
sum(exp(log_dens(G))) * diff(g1)[1] * diff(g2)[1]
