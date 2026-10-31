# Cap. 14: Testes na fronteira do espaço paramétrico
source(file.path("R", "comum", "gerador.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
# (a) Parâmetro com sinal conhecido: t = 2, nu = 20
tt <- 2; nu <- 20
pr("bilateral %.4f unilateral %.4f",
   2 * pt(tt, nu, lower.tail = FALSE), pt(tt, nu, lower.tail = FALSE))
pr("mistura com F %.4f mistura com chi2 %.4f",
   0.5 * pf(tt^2, 1, nu, lower.tail = FALSE),
   0.5 * pchisq(tt^2, 1, lower.tail = FALSE))
# (b) Componente de variância: a grupos de n; LR por ML
# O quantil c95 sai com um decimal: é o que o erro de Monte Carlo
# permite com N = 200000 (erro-padrão de cerca de 0,02)
rv <- function(sqb, sqd, a, n) {
  Fv <- (sqb / (a - 1)) / (sqd / (a * (n - 1)))
  sqt <- sqb + sqd
  ifelse(Fv <= a / (a - 1), 0,
    a * n * log(sqt / (a * n)) - a * (n - 1) *
      log(sqd / (a * (n - 1))) - a * log(sqb / a))
}
pr("%3s %3s %8s %8s %8s %8s", "a", "n", "P(RV=0)", "t(chi2)",
   "t(mist)", "c95")
N <- 200000
for (cfg in list(c(6, 3), c(10, 4), c(30, 3), c(200, 3))) {
  a <- cfg[1]; n <- cfg[2]; g <- gerador(1987)
  sqb <- qchisq(g$unif(N), a - 1)
  sqd <- qchisq(g$unif(N), a * (n - 1))
  L <- rv(sqb, sqd, a, n)
  pr("%3d %3d %8.4f %8.4f %8.4f %8.1f", a, n,
     pf(a / (a - 1), a - 1, a * (n - 1)), mean(L > qchisq(0.95, 1)),
     mean(L > qchisq(0.90, 1)), sort(L)[0.95 * N])
}
# (c) Alcance da correlação na aveia: lei exata de RV por simulação.
# RV não depende de beta nem de sigma: basta simular ruído branco.
av <- read.csv(file.path("dados", "yates_oats.csv"))
y <- av$yield; n <- length(y); link <- 0.2012
X <- model.matrix(~ factor(block) + factor(nitro) + factor(gen), av)
D <- as.matrix(dist(cbind((av$col - 1) * 44 * link,
                          (av$row - 1) * 28.4 * link)))
dimnames(D) <- NULL
fases <- c(0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 29, 35, 45, 60)
pre <- lapply(fases, function(f) {
  R <- if (f == 0) diag(n) else exp(-D / f)
  Ri <- solve(R); RiX <- Ri %*% X
  A <- Ri - RiX %*% solve(crossprod(X, RiX), t(RiX))
  list(A = (A + t(A)) / 2,
       ld = as.numeric(determinant(R)$modulus))
})
perf <- function(Y) sapply(pre, function(p)
  -n / 2 * log(rowSums((Y %*% p$A) * Y) / n) - p$ld / 2)
obs <- drop(perf(rbind(y)))
rvobs <- 2 * (max(obs) - obs[1])
N <- 20000; g <- gerador(2014)
Z <- matriz_normal(g, N, n)
L <- perf(Z); rv0 <- 2 * (apply(L, 1, max) - L[, 1])
pr("RV obs %.3f em phi = %g", rvobs, fases[which.max(obs)])
pr("P(RV<0.01) %.4f c95 %.3f max %.3f p sim %.5f", mean(rv0 < 0.01),
   sort(rv0)[0.95 * N], max(rv0), (1 + sum(rv0 >= rvobs)) / (N + 1))
pr("p chi2 %.5f p mistura %.5f",
   pchisq(rvobs, 1, lower.tail = FALSE),
   0.5 * pchisq(rvobs, 1, lower.tail = FALSE))
# Por que tanta massa em zero: os resíduos de MQ de parcelas vizinhas
# já são negativamente correlacionados (blocos no modelo)
W <- (D > 0 & D < 9) * 1
M <- diag(n) - X %*% solve(crossprod(X), t(X))
pr("correlação de vizinhos em média sob H0 %.3f (obs. %.3f)",
   n * sum(M * W) / ((n - ncol(X)) * sum(W)),
   n * sum((M %*% y) * (W %*% M %*% y)) / (sum(W) * sum((M %*% y)^2)))
