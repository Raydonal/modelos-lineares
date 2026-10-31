# Cap. 14: Comparações múltiplas: multiplicadores e cobertura
source(file.path("R", "comum", "gerador.R"))
source(file.path("R", "comum", "multiplas.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
mult <- function(a, nu, al = 0.05) {
  m <- a * (a - 1) / 2
  c(t = qt(1 - al / 2, nu), bonf = qt(1 - al / (2 * m), nu),
    tukey = qtukey(1 - al, a, nu) / sqrt(2),
    dunnett = q_dunnett(a - 1, nu),
    scheffe = sqrt((a - 1) * qf(1 - al, a - 1, nu)))
}
pr("%3s %3s %6s %6s %6s %6s %6s", "a", "nu", "t", "Bonf", "Tukey",
   "Dunn", "Sch")
for (a in c(3, 5, 10, 20)) {
  v <- mult(a, 4 * a)
  pr("%3d %3d %6.3f %6.3f %6.3f %6.3f %6.3f", a, 4 * a, v[1], v[2],
     v[3], v[4], v[5])
}
# Exemplo: 3 grupos de 5, s = 2, médias 10, 12, 15
v <- mult(3, 12); e3 <- 2 * sqrt(2 / 5)
pr("ep %.4f limites: t %.3f Bonf %.3f Tukey %.3f Dunn %.3f Sch %.3f",
   e3, v[1] * e3, v[2] * e3, v[3] * e3, v[4] * e3, v[5] * e3)
# Cobertura simultânea das diferenças de pares: a = 5, r = 5
a <- 5; r <- 5; nu <- a * (r - 1); N <- 20000
g <- gerador(1414); v <- mult(a, nu); ep <- sqrt(2 / r)
Z <- matriz_normal(g, N, a * r)
ok <- matrix(0, N, 4)
for (i in seq_len(N)) {
  Y <- matrix(Z[i, ], a, r, byrow = TRUE)
  md <- rowMeans(Y); s <- sqrt(sum((Y - md)^2) / nu)
  mx <- (max(md) - min(md)) / (s * ep)
  ok[i, ] <- mx <= v[c(1, 2, 3, 5)]
}
cob <- colMeans(ok)
pr("cobertura: t %.4f Bonf %.4f Tukey %.4f Scheffé %.4f", cob[1],
   cob[2], cob[3], cob[4])
# Benjamini-Hochberg e Holm à mão
p <- c(0.001, 0.012, 0.024, 0.035, 0.6); m <- 5; al <- 0.05
pr("limiares BH %s", paste(sprintf("%.3f", al * (1:m) / m),
                           collapse = " "))
holm <- 0
while (holm < m && p[holm + 1] <= al / (m - holm)) holm <- holm + 1
pr("rejeições: Bonferroni %d Holm %d BH %d", sum(p <= al / m), holm,
   bh(p, al))
