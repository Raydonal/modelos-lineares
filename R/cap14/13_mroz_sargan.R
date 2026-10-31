# Cap. 14: Mroz: sobreidentificação dos instrumentos
pr <- function(...) cat(sprintf(...), "\n", sep = "")
m <- read.csv(file.path("dados", "mroz.csv"))
y <- m$lwage; n <- length(y); um <- rep(1, n)
proj <- function(Z, v) Z %*% solve(crossprod(Z), crossprod(Z, v))
fit <- function(X, Z) {
  Xh <- proj(Z, X); A <- solve(crossprod(Xh, X))
  b <- drop(A %*% crossprod(Xh, y)); e <- y - X %*% b
  S <- sum(proj(Z, e)^2) / (sum(e^2) / n)
  gl <- ncol(Z) - ncol(X)
  ep <- sqrt(diag(A * sum(e^2) / (n - ncol(X))))
  list(b = b, ep = ep, S = S, gl = gl,
       p = if (gl > 0) pchisq(S, gl, lower.tail = FALSE) else NA)
}
# Primeiro estágio: teste F de pai e mãe na escolaridade
fse <- function(W) {
  x <- m$educ
  r0 <- x - proj(W, x); Z <- cbind(W, m$fatheduc, m$motheduc)
  r1 <- x - proj(Z, x); gl <- n - ncol(Z)
  Fv <- ((sum(r0^2) - sum(r1^2)) / 2) / (sum(r1^2) / gl)
  pr("primeiro estágio: F(2,%d) %.2f", gl, Fv)
}
ex <- cbind(m$exper, m$expersq)
X1 <- cbind(um, m$educ); X2 <- cbind(um, m$educ, ex)
for (k in 1:2) {
  X <- if (k == 1) X1 else X2
  W <- if (k == 1) cbind(um) else cbind(um, ex)
  pr("modelo %d", k); fse(W)
  mq <- solve(crossprod(X), crossprod(X, y))[2]
  fp <- fit(X, cbind(W, m$fatheduc))
  fm <- fit(X, cbind(W, m$motheduc))
  f2 <- fit(X, cbind(W, m$fatheduc, m$motheduc))
  pr("MQ %.4f pai %.4f (%.4f) mãe %.4f (%.4f)", mq, fp$b[2], fp$ep[2],
     fm$b[2], fm$ep[2])
  pr("ambos %.4f (%.4f)", f2$b[2], f2$ep[2])
  pr("Sargan S %.4f gl %d p %.3f", f2$S, f2$gl, f2$p)
}
