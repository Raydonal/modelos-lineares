# Cap. 6 -- Gradiente descendente: taxa de aprendizado, autovalores da hessiana e padronização
db <- read.csv(file.path("dados", "diabetes.csv"))
y <- db$y; n <- length(y)
X_bruto <- cbind(1, as.matrix(db[, 1:10]))
X_pad <- cbind(1, scale(db[, 1:10]))
# A hessiana da perda f(b) = ||y - X b||^2 / (2n) é H = X'X / n
espectro <- function(X) { ev <- eigen(crossprod(X) / n, symmetric = TRUE)$values; c(max = max(ev), min = min(ev), kappa = max(ev) / min(ev)) }
signif(rbind(bruto = espectro(X_bruto), padronizado = espectro(X_pad)), 4)

gd <- function(X, eta, iter) {
  bs <- solve(crossprod(X), crossprod(X, y))           # mínimo exato
  b <- rep(0, ncol(X)); d <- numeric(iter)
  for (k in 1:iter) { b <- b - eta * drop(crossprod(X, X %*% b - y)) / n; d[k] <- sqrt(sum((b - bs)^2)) }
  d
}
ev <- eigen(crossprod(X_pad) / n, symmetric = TRUE)$values
L <- max(ev); mu <- min(ev)
etas <- c(meio = 1 / L, otimo = 2 / (L + mu), quase_2 = 1.98 / L, acima_2 = 2.02 / L)
R <- sapply(etas, function(eta) gd(X_pad, eta, 500))
signif(R[c(1, 10, 100, 500), ], 4)
# Fator de contração previsto, max |1 - eta lambda_i|, e o observado no fim
fator <- sapply(etas, function(eta) max(abs(1 - eta * ev)))
observado <- (R[500, ] / R[400, ])^(1 / 100)
round(rbind(previsto = fator, observado = observado), 5)
# Iterações para reduzir o erro por 10^6, com o passo ótimo
kap <- L / mu
c(kappa = kap, iteracoes = log(1e6) / -log((kap - 1) / (kap + 1)))
