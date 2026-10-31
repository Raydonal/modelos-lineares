# Cap. 7 -- Número de condição: sistemas lineares e mínimos quadrados
kappa2 <- function(M) { s <- svd(M)$d; max(s) / min(s) }
H <- outer(1:8, 1:8, function(i, j) 1 / (i + j - 1))      # matriz de Hilbert 8 x 8
x <- rep(1, 8); b <- H %*% x
c(kappa_hilbert = kappa2(H), erro_relativo_solve = sqrt(sum((solve(H, b) - x)^2)) / sqrt(8))
# Perturbação no lado direito: a cota kappa * ||db|| / ||b|| é atingida na pior direção
s <- svd(H); db <- 1e-10 * s$u[, 8]                       # perturbação na pior direção
bb <- s$d[1] * s$u[, 1]                                    # lado direito na direção mais "esticada"
xs <- s$v[, 1]                                             # solução de H x = bb
dx <- solve(H, bb + db) - xs
c(amplificacao = (sqrt(sum(dx^2)) / sqrt(sum(xs^2))) / (sqrt(sum(db^2)) / sqrt(sum(bb^2))), kappa = kappa2(H))
# Mínimos quadrados: equações normais perdem o dobro de dígitos que a QR
source(file.path("R", "comum", "gerador.R")); g <- gerador(75)
U <- qr.Q(qr(matriz_normal(g, 50, 5))); V <- qr.Q(qr(matriz_normal(g, 5, 5)))
res <- sapply(c(1e2, 1e4, 1e6, 1e7), function(k) {
  X <- U %*% diag(k^-(0:4 / 4)) %*% t(V)                   # kappa(X) = k
  bet <- rep(1, 5); y <- X %*% bet
  b_ne <- solve(crossprod(X), crossprod(X, y)); b_qr <- qr.coef(qr(X), y)
  c(kappa = kappa2(X), erro_normais = max(abs(b_ne - bet)), erro_qr = max(abs(b_qr - bet)))
})
signif(res, 3)
