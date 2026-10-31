# Cap. 8 -- IS-LM não linear: estática comparativa pelo teorema da função implícita
# IS: Y = c0 + c1 (Y - T) + I(r) + G, com I(r) = 800 (r/5)^(-0,3125)
# LM: M/P = L(Y, r) = 500 (Y/4000)^2 (r/5)^(-1)      (tangentes ao modelo linear do Cap. 3 em Y = 4000, r = 5)
c0 <- 100; c1 <- 0.75; Tx <- 200
Fsis <- function(x, G, M) { Y <- x[1]; r <- x[2]
  c(Y - c0 - c1 * (Y - Tx) - 800 * (r / 5)^(-0.3125) - G, M - 500 * (Y / 4000)^2 * (r / 5)^(-1)) }
Jx <- function(x) { Y <- x[1]; r <- x[2]                    # jacobiano em relação a (Y, r)
  rbind(c(1 - c1, 0.3125 * 800 / 5 * (r / 5)^(-1.3125)),
        c(-1000 / 4000 * (Y / 4000) * (r / 5)^(-1), 500 * (Y / 4000)^2 * (r / 5)^(-2) / 5)) }
Jtheta <- rbind(c(-1, 0), c(0, 1))                           # jacobiano em relação a (G, M)
# Newton em (Y, log r), que mantém os juros positivos: d/d(log r) = r d/dr
newton <- function(G, M, x = c(4000, 5)) {
  z <- c(x[1], log(x[2]))
  for (k in 1:100) { x <- c(z[1], exp(z[2])); Jz <- Jx(x) %*% diag(c(1, x[2]))
    z <- z - solve(Jz, Fsis(x, G, M)) }
  c(z[1], exp(z[2]))
}
x0 <- newton(250, 500); x0
mult <- -solve(Jx(x0), Jtheta); dimnames(mult) <- list(c("Y", "r"), c("dG", "dM")); round(mult, 5)
# Conferência por diferenças finitas
h <- 1e-4
round(cbind(dG = (newton(250 + h, 500) - newton(250 - h, 500)) / (2 * h),
            dM = (newton(250, 500 + h) - newton(250, 500 - h)) / (2 * h)), 5)
# Expansão monetária: o modelo linear do Cap. 3 levaria os juros a zero; no não linear, o efeito se esgota
res <- t(sapply(c(500, 750, 1000, 1500, 2000, 3000), function(M) {
  x <- newton(250, M, x = c(4000, 5)); m <- -solve(Jx(x), Jtheta)
  c(M = M, Y = x[1], r = x[2], dY_dM = m[1, 2], Y_linear = 4000 + (M - 500) * 50 / 37.5, r_linear = 5 - (M - 500) * 0.25 / 37.5)
}))
round(res, 3)
