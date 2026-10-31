# Cap. 13 -- Mínimos quadrados restritos: retornos constantes
source(file.path("R", "comum", "gerador.R")); g <- gerador(1928)
n <- 30                                       # 30 firmas simuladas
lK <- 2 + g$normal(n); lL <- 3 + 0.8 * lK + 0.6 * g$normal(n)
lQ <- 1 + 0.3 * lK + 0.7 * lL + 0.4 * g$normal(n)  # alfa + gama = 1
X <- cbind(1, lK, lL); A <- solve(crossprod(X))
b <- drop(A %*% crossprod(X, lQ))             # irrestrito
K <- c(0, 1, 1); m <- 1                       # restrição K'b = m
bK <- b - A %*% K %*% solve(t(K) %*% A %*% K, sum(K * b) - m)
# Conferência: impor a restrição reescrevendo o modelo
r <- coef(lm(I(lQ - lL) ~ I(lK - lL)))
round(rbind(irrestrito = b, restrito = drop(bK),
            reescrito = c(r, 1 - r[2])), 4)
s2 <- sum((lQ - X %*% b)^2) / (n - 3)
CK <- A - A %*% K %*% t(K) %*% A / drop(t(K) %*% A %*% K)
round(rbind(ep_irrestrito = sqrt(s2 * diag(A)),
            ep_restrito = sqrt(s2 * diag(CK))), 4)
# Aumento da SQRes = (K'b - m)^2 / K'AK
c(dif_SQRes = sum((lQ - X %*% bK)^2) - sum((lQ - X %*% b)^2),
  formula = (sum(K * b) - m)^2 / drop(t(K) %*% A %*% K))
