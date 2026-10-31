# Cap. 14: Somas de quadrados sequenciais e parciais
pr <- function(...) cat(sprintf(...), "\n", sep = "")
y <- c(3, 1, -2, -2); n <- 4
um <- rep(1, 4); x1 <- c(1, 1, -1, -1); x2 <- c(1, 0, 0, -1)
sqr <- function(...) {
  Z <- cbind(...)
  sum((y - Z %*% solve(crossprod(Z), crossprod(Z, y)))^2)
}
t0 <- sqr(um); r1 <- sqr(um, x1); r2 <- sqr(um, x2)
r12 <- sqr(um, x1, x2)
pr("SQ total %.2f residual %.2f", t0, r12)
pr("x1 primeiro: %.2f + %.2f", t0 - r1, r1 - r12)
pr("x2 primeiro: %.2f + %.2f", t0 - r2, r2 - r12)
pr("parciais: x1 %.2f x2 %.2f (soma %.2f)", r2 - r12, r1 - r12,
   r2 - r12 + r1 - r12)
# Parcial = t^2 s^2 = b^2 / [(X'X)^-1]_jj
X <- cbind(um, x1, x2); A <- solve(crossprod(X))
b <- drop(A %*% crossprod(X, y))
pr("b %.3f %.3f %.3f", b[1], b[2], b[3])
pr("b1^2/A11 %.2f b2^2/A22 %.2f", b[2]^2 / A[2, 2],
   b[3]^2 / A[3, 3])
# Testes: sequencial e parcial com o mesmo denominador
for (s in c(16, 4.5, 12.5, 1)) pr("SQ %5.2f F %5.2f p %.3f", s,
  s / r12, pf(s / r12, 1, 1, lower.tail = FALSE))
