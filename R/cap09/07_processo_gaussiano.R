# Cap. 9: Regressão por processo gaussiano (mcycle)
mc <- read.csv(file.path("dados", "mcycle.csv"))
t <- mc$times; y <- mc$accel; n <- length(y)
r <- y - mean(y)
nucleo <- function(a, b, l) exp(-outer(a, b, "-")^2 / (2 * l^2))

# Verossimilhança perfilada: Var(y) = s2 (C_l + tau I),
# com s2 de forma fechada
perfil <- function(l, tau) {
  R <- chol(nucleo(t, t, l) + tau * diag(n))     # A = R'R
  u <- forwardsolve(t(R), r); s2 <- sum(u^2) / n
  -n / 2 * log(2 * pi * s2) - sum(log(diag(R))) - n / 2
}
grade <- expand.grid(l = seq(1, 10, by = 0.25),
                     tau = 10^seq(-2, 1, by = 0.05))
grade$ll <- mapply(perfil, grade$l, grade$tau)
o <- grade[which.max(grade$ll), ]; l <- o$l; tau <- o$tau

A <- nucleo(t, t, l) + tau * diag(n)
s2 <- drop(t(r) %*% solve(A, r)) / n
round(c(l = l, tau = tau, sigma_f = sqrt(s2), sigma = sqrt(s2 * tau),
        logver = o$ll, gl = sum(diag(solve(A, nucleo(t, t, l))))), 3)

# Posterior de f nos tempos t* (condicional da normal conjunta)
ts <- c(10, 15, 20, 30, 40, 50)
Cs <- nucleo(ts, t, l)
media <- mean(y) + drop(Cs %*% solve(A, r))
dp <- sqrt(s2 * (1 - rowSums(Cs * t(solve(A, t(Cs))))))
round(rbind(t = ts, media = media, dp = dp), 2)
