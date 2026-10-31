# Cap. 14: Regiões de confiança e intervalos de predição
source(file.path("R", "comum", "gerador.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
# Elipse para as duas médias: (2, 5), s2 = 4/3, gl = 3
F23 <- qf(0.95, 2, 3); s2 <- 4 / 3
pr("F(2,3) %.3f c %.3f", F23, 2 * s2 * F23)
pr("semieixos %.3f %.3f", sqrt(2 * s2 * F23 / 2),
   sqrt(2 * s2 * F23 / 3))
for (v in list(c(3, 3), c(0, 0)))
  pr("(%g, %g): forma %.1f", v[1], v[2],
     2 * (v[1] - 2)^2 + 3 * (v[2] - 5)^2)
# Sombra da elipse: Scheffé contra t para cada média
t3 <- qt(0.975, 3)
for (nj in c(2, 3))
  pr("n %d: t %.3f  sombra %.3f", nj, t3 * sqrt(s2 / nj),
     sqrt(2 * F23) * sqrt(s2 / nj))
# Três pontos: média e nova observação em x0 = 3
x <- 0:2; y <- c(1, 2, 4); X <- cbind(1, x)
A <- solve(crossprod(X)); b <- drop(A %*% crossprod(X, y))
s2 <- sum((y - X %*% b)^2); x0 <- c(1, 3)
y0 <- sum(x0 * b); h <- drop(t(x0) %*% A %*% x0); tt <- qt(0.975, 1)
pr("y0 %.4f h %.4f t %.3f", y0, h, tt)
pr("IC +- %.3f  IP +- %.3f", tt * sqrt(s2 * h),
   tt * sqrt(s2 * (1 + h)))
# Cobertura: regressão com 30 pontos, erros de variância 1
X <- cbind(1, seq(0, 1, length.out = 30)); A <- solve(crossprod(X))
H <- A %*% t(X); M <- diag(30) - X %*% H; x0 <- c(1, 1.2)
h <- drop(t(x0) %*% A %*% x0); tt <- qt(0.975, 28)
g <- gerador(1405); N <- 20000
erro <- list(
  normal = function(k) g$normal(k),
  uniforme = function(k) sqrt(12) * (g$unif(k) - 0.5),
  exponencial = function(k) -log(g$unif(k)) - 1)
pr("%-12s %7s %7s %7s", "erro", "cob.IC", "falhaE", "falhaD")
for (nm in names(erro)) {
  E <- matrix(erro[[nm]](30 * N), N, 30, byrow = TRUE)
  e0 <- erro[[nm]](N)
  dif <- drop(E %*% t(H) %*% x0)            # ajuste - média
  s <- sqrt(rowSums((E %*% M)^2) / 28)
  ic <- mean(abs(dif) <= tt * s * sqrt(h))
  z <- (e0 - dif) / (s * sqrt(1 + h))       # y0 - ajuste
  pr("%-12s %7.4f %7.4f %7.4f", nm, ic, mean(z < -tt), mean(z > tt))
}
