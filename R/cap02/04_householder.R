# Cap. 2 -- Fatoração QR por reflexões de Householder
householder_qr <- function(A) {
  m <- nrow(A); n <- ncol(A); R <- A; Q <- diag(m)
  for (k in 1:min(n, m - 1)) {
    x <- R[k:m, k]
    alfa <- -sign(x[1] + (x[1] == 0)) * sqrt(sum(x^2))  # sinal que evita cancelamento
    v <- x; v[1] <- v[1] - alfa                          # v = x - alfa e1
    if (sum(v^2) == 0) next
    v <- v / sqrt(sum(v^2))
    H <- diag(m); H[k:m, k:m] <- diag(m - k + 1) - 2 * v %o% v
    R <- H %*% R                                         # zera a coluna k abaixo da diagonal
    Q <- Q %*% H                                         # Q = H1 H2 ... Hn
  }
  R[abs(R) < 1e-14] <- 0
  list(Q = Q, R = R)
}

X <- cbind(1, c(1, 2, 3, 4, 5), c(1, 4, 9, 16, 25))       # regressão quadrática
f <- householder_qr(X)
round(f$R, 4)                                            # 5 x 3, triangular superior
max(abs(f$Q %*% f$R - X)); max(abs(crossprod(f$Q) - diag(5)))
round(qr.R(qr(X)), 4)                                    # R do qr() (sinais podem diferir)
