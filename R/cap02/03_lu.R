# Cap. 2: Fatoração LU com e sem pivotamento; substituições progressiva e regressiva
lu_sem_pivo <- function(A) {
  n <- nrow(A); L <- diag(n); U <- A
  for (k in 1:(n - 1)) {
    if (U[k, k] == 0) stop("pivô nulo na etapa ", k)
    for (i in (k + 1):n) {
      L[i, k] <- U[i, k] / U[k, k]              # multiplicador
      U[i, ] <- U[i, ] - L[i, k] * U[k, ]       # elimina abaixo do pivô
    }
  }
  list(L = L, U = U)
}
lu_pivo <- function(A) {                        # PA = LU
  n <- nrow(A); L <- diag(n); U <- A; p <- 1:n
  for (k in 1:(n - 1)) {
    q <- which.max(abs(U[k:n, k])) + k - 1      # maior pivô disponível
    U[c(k, q), ] <- U[c(q, k), ]; p[c(k, q)] <- p[c(q, k)]
    if (k > 1) L[c(k, q), 1:(k - 1)] <- L[c(q, k), 1:(k - 1)]
    for (i in (k + 1):n) {
      L[i, k] <- U[i, k] / U[k, k]
      U[i, ] <- U[i, ] - L[i, k] * U[k, ]
    }
  }
  list(P = diag(n)[p, ], L = L, U = U)
}
progressiva <- function(L, b) {                 # resolve Lz = b, L triangular inferior
  z <- numeric(length(b))
  for (i in seq_along(b)) z[i] <- (b[i] - sum(L[i, seq_len(i - 1)] * z[seq_len(i - 1)])) / L[i, i]
  z
}
regressiva <- function(U, z) {                  # resolve Ux = z, U triangular superior
  n <- length(z); x <- numeric(n)
  for (i in n:1) {
    j <- if (i < n) (i + 1):n else integer(0)
    x[i] <- (z[i] - sum(U[i, j] * x[j])) / U[i, i]
  }
  x
}

A <- matrix(c(2, 1, 1, 0,
              4, 3, 3, 1,
              8, 7, 9, 5,
              6, 7, 9, 8), 4, 4, byrow = TRUE)
f <- lu_sem_pivo(A); f$L; f$U
max(abs(f$L %*% f$U - A))
b <- c(1, 2, 3, 4)
x <- regressiva(f$U, progressiva(f$L, b))
x; max(abs(A %*% x - b))

# Um pivô minúsculo: sem pivotamento o resultado é destruído pelo arredondamento
E <- matrix(c(1e-20, 1, 1, 1), 2, 2, byrow = TRUE); bE <- c(1, 2)
g <- lu_sem_pivo(E); h <- lu_pivo(E)
rbind(sem_pivo = regressiva(g$U, progressiva(g$L, bE)),
      com_pivo = regressiva(h$U, progressiva(h$L, h$P %*% bE)),
      solve    = solve(E, bE))
