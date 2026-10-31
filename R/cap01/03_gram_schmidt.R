# Cap. 1 -- Processo de Gram-Schmidt (versão clássica)
gram_schmidt <- function(X) {
  Q <- matrix(0, nrow(X), ncol(X))
  for (j in seq_len(ncol(X))) {
    v <- X[, j]
    for (i in seq_len(j - 1)) {
      v <- v - sum(Q[, i] * X[, j]) * Q[, i]   # remove a componente em u_i
    }
    if (sqrt(sum(v^2)) < 1e-12) stop("colunas linearmente dependentes")
    Q[, j] <- v / sqrt(sum(v^2))               # normaliza
  }
  Q
}

X <- cbind(c(1, 1, 1, 1), c(1, 2, 3, 4), c(1, 4, 9, 16))   # 1, t, t^2
Q <- gram_schmidt(X)
round(Q, 4)
round(crossprod(Q), 12)          # Q'Q = I
R <- crossprod(Q, X)             # R = Q'X é triangular superior
round(R, 4)
max(abs(Q %*% R - X))            # X = QR
