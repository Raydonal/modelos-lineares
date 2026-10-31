# Cap. 14: Salários: falta de ajuste e somas de quadrados
pr <- function(...) cat(sprintf(...), "\n", sep = "")
w <- read.csv(file.path("dados", "wage1.csv"))
y <- log(w$wage); e <- w$educ; n <- length(y)
um <- rep(1, n); ej <- sort(unique(e)); c0 <- length(ej)
D <- outer(e, ej, "==") * 1
sqr <- function(...) sum(qr.resid(qr(cbind(...)), y)^2)
pur <- sqr(D); glp <- n - c0
pr("caselas %d SQ puro %.3f gl %d QM %.4f", c0, pur, glp, pur / glp)
lof <- function(nome, ...) {
  s <- sqr(...); r <- qr(cbind(...))$rank
  Fv <- ((s - pur) / (c0 - r)) / (pur / glp)
  pr("%s: SQ falta %.3f gl %d F %.2f p %.2e", nome, s - pur, c0 - r,
     Fv, pf(Fv, c0 - r, glp, lower.tail = FALSE))
}
q12 <- pmax(e - 12, 0)
lof("reta", um, e); lof("parábola", um, e, e^2)
lof("quebrada", um, e, q12)
# Dois encaixados dentro do modelo: a quebra é necessária?
s1 <- sqr(um, e); s2 <- sqr(um, e, q12)
Fq <- (s1 - s2) / (s2 / (n - 3))
pr("quebra: F(1,%d) %.2f p %.1e", n - 3, Fq,
   pf(Fq, 1, n - 3, lower.tail = FALSE))
# Só com caselas de 8 anos ou mais (poucos dados abaixo)
ok <- e >= 8
pr("caselas abaixo de 8 anos: %d pessoas", sum(!ok))
# Somas sequenciais e parciais: mulher, educ, exper, exper2, tenure
f <- w$female; x <- w$exper; t <- w$tenure
cols <- list(mulher = f, educ = e, exper = cbind(x, x^2), tenure = t)
gl <- c(1, 1, 2, 1)
tudo <- do.call(cbind, c(list(um), cols)); glr <- n - ncol(tudo)
sr <- sum(qr.resid(qr(tudo), y)^2); qmr <- sr / glr
pr("SQRes %.3f gl %d QM %.4f", sr, glr, qmr)
seqs <- function(ordem) {
  s0 <- sqr(um); Z <- um; out <- numeric(4)
  for (nm in ordem) {
    Z <- cbind(Z, cols[[nm]]); s <- sqr(Z)
    out[match(nm, names(cols))] <- s0 - s; s0 <- s
  }
  out
}
a <- seqs(names(cols)); b <- seqs(rev(names(cols)))
pc <- sapply(seq_along(cols), function(j)
  sum(qr.resid(qr(do.call(cbind, c(list(um), cols[-j]))), y)^2) - sr)
pr("%-7s %2s %8s %8s %8s %7s", "termo", "gl", "direta", "inversa",
   "parcial", "F parc")
for (j in 1:4) pr("%-7s %2d %8.3f %8.3f %8.3f %7.2f", names(cols)[j],
  gl[j], a[j], b[j], pc[j], pc[j] / gl[j] / qmr)
pr("somas: %.3f %.3f %.3f (total %.3f)", sum(a), sum(b), sum(pc),
   sqr(um) - sr)
# Hiato de gênero: bruto e ajustado
br <- solve(crossprod(cbind(um, f)), crossprod(cbind(um, f), y))
A <- solve(crossprod(tudo)); bt <- drop(A %*% crossprod(tudo, y))
pr("hiato bruto %.3f ajustado %.3f ep %.3f", br[2], bt[2],
   sqrt(qmr * A[2, 2]))
# Intervalos em educ = 16 no modelo da reta quebrada
X <- cbind(um, e, q12); A <- solve(crossprod(X))
b <- drop(A %*% crossprod(X, y)); s2 <- sum((y - X %*% b)^2) / (n - 3)
x0 <- c(1, 16, 4); y0 <- sum(x0 * b); h <- drop(t(x0) %*% A %*% x0)
tq <- qt(0.975, n - 3); mw <- sqrt(3 * qf(0.95, 3, n - 3))
pr("educ 16: log %.3f IC +- %.3f IP +- %.3f WH +- %.3f", y0,
   tq * sqrt(s2 * h), tq * sqrt(s2 * (1 + h)), mw * sqrt(s2 * h))
pr("salário: mediana %.2f IP (%.2f, %.2f)", exp(y0),
   exp(y0 - tq * sqrt(s2 * (1 + h))),
   exp(y0 + tq * sqrt(s2 * (1 + h))))
