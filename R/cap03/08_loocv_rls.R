# Cap. 3 -- Validação cruzada leave-one-out em forma fechada e mínimos quadrados recursivos
db <- read.csv(file.path("dados", "diabetes.csv"))
X <- cbind(1, as.matrix(db[, 1:10])); y <- db$y
n <- nrow(X); p <- ncol(X)

f <- qr(X)
e <- qr.resid(f, y)                              # resíduos
h <- rowSums(qr.Q(f)^2)                          # alavancas h_ii = ||linha i de Q1||^2
loo_fechado <- e / (1 - h)                       # Sherman-Morrison

t_bruto <- system.time(loo_bruto <- sapply(1:n, function(i) {
  b <- qr.solve(X[-i, ], y[-i]); y[i] - sum(X[i, ] * b)
}))[["elapsed"]]
c(max_dif = max(abs(loo_fechado - loo_bruto)), segundos_bruto = t_bruto,
  PRESS = sum(loo_fechado^2), SQres = sum(e^2))
summary(h)

# Mínimos quadrados recursivos: começa com as 20 primeiras observações
m0 <- 20
P <- solve(crossprod(X[1:m0, ])); b <- P %*% crossprod(X[1:m0, ], y[1:m0])
for (t in (m0 + 1):n) {
  x <- X[t, ]
  Px <- P %*% x
  P <- P - Px %*% t(Px) / as.numeric(1 + crossprod(x, Px))   # Sherman-Morrison
  b <- b + P %*% x * as.numeric(y[t] - crossprod(x, b))       # atualização do coeficiente
}
max(abs(b - qr.coef(f, y)))                      # coincide com o MQO em lote
