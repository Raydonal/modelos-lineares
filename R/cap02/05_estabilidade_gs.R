# Cap. 2 -- Perda de ortogonalidade: Gram-Schmidt clássico, modificado e Householder
gs_classico <- function(X) {
  Q <- X
  for (j in seq_len(ncol(X))) {
    v <- X[, j]
    if (j > 1) v <- v - Q[, 1:(j - 1), drop = FALSE] %*% crossprod(Q[, 1:(j - 1), drop = FALSE], X[, j])
    Q[, j] <- v / sqrt(sum(v^2))
  }
  Q
}
gs_modificado <- function(X) {
  Q <- X
  for (j in seq_len(ncol(X))) {
    Q[, j] <- Q[, j] / sqrt(sum(Q[, j]^2))
    if (j < ncol(X)) for (k in (j + 1):ncol(X))
      Q[, k] <- Q[, k] - sum(Q[, j] * Q[, k]) * Q[, j]   # ortogonaliza o restante contra q_j
  }
  Q
}
# matriz 60 x 12 com número de condição kappa prescrito
matriz_cond <- function(kappa, m = 60, n = 12) {
  set.seed(1)
  U <- qr.Q(qr(matrix(rnorm(m * n), m, n)))
  V <- qr.Q(qr(matrix(rnorm(n * n), n, n)))
  U %*% diag(kappa^(-(0:(n - 1)) / (n - 1))) %*% t(V)
}
perda <- function(Q) max(abs(crossprod(Q) - diag(ncol(Q))))
tab <- t(sapply(10^c(2, 5, 8, 11), function(k) {
  X <- matriz_cond(k)
  c(kappa = k, classico = perda(gs_classico(X)), modificado = perda(gs_modificado(X)),
    householder = perda(qr.Q(qr(X, LAPACK = TRUE))))
}))
signif(tab, 2)
