# Cap. 14: Razão de verossimilhanças, Wald e escore
pr <- function(...) cat(sprintf(...), "\n", sep = "")
est <- function(Fv, q, n, r) {
  x <- q * Fv / (n - r)
  c(W = n * x, RV = n * log1p(x), E = n * x / (1 + x))
}
# Os dois grupos: F = 8,1 com n = 5, r = 2, q = 1
s <- est(8.1, 1, 5, 2)
pr("W %.3f RV %.3f E %.3f", s[1], s[2], s[3])
pr("p W %.4f RV %.4f E %.4f F %.4f",
   pchisq(s[1], 1, lower.tail = FALSE),
   pchisq(s[2], 1, lower.tail = FALSE),
   pchisq(s[3], 1, lower.tail = FALSE),
   pf(8.1, 1, 3, lower.tail = FALSE))
# Conta direta a partir dos resíduos
y <- c(1, 3, 4, 6, 5); e1 <- y - c(2, 2, 5, 5, 5); e0 <- y - mean(y)
s1 <- sum(e1^2) / 5; s0 <- sum(e0^2) / 5
X <- cbind(c(1, 1, 0, 0, 0), c(0, 0, 1, 1, 1))
P <- X %*% solve(crossprod(X), t(X))
pr("s1 %.3f s0 %.3f W %.3f RV %.3f E %.3f", s1, s0,
   (sum(e0^2) - sum(e1^2)) / s1, 5 * log(s0 / s1),
   sum(e0 * (P %*% e0)) / s0)
# Tamanho real a 5% (calibração qui-quadrado): exato, pois cada
# estatística é função crescente de F
tam <- function(q, n, r) {
  cc <- qchisq(0.95, q); nu <- n - r
  c(W = pf(cc * nu / (n * q), q, nu, lower.tail = FALSE),
    RV = pf(expm1(cc / n) * nu / q, q, nu, lower.tail = FALSE),
    E = if (cc < n) pf(cc / (1 - cc / n) * nu / (n * q), q, nu,
                       lower.tail = FALSE) else 0)
}
pr("%5s %5s %7s %7s %7s", "n", "r", "W", "RV", "E")
for (n in c(10, 20, 50, 100, 500)) for (q in 3) {
  r <- 5; t <- tam(q, n, r)
  pr("%5d %5d %7.3f %7.3f %7.3f", n, r, t[1], t[2], t[3])
}
