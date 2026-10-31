# Cap. 3 -- Inversa por Gauss-Jordan, pela adjunta e regra de Cramer
gauss_jordan_inv <- function(A) {
  n <- nrow(A); M <- cbind(A, diag(n))          # [A | I]
  for (j in 1:n) {
    k <- which.max(abs(M[j:n, j])) + j - 1      # pivotamento parcial
    if (abs(M[k, j]) < 1e-12) stop("matriz singular")
    M[c(j, k), ] <- M[c(k, j), ]
    M[j, ] <- M[j, ] / M[j, j]
    for (i in setdiff(1:n, j)) M[i, ] <- M[i, ] - M[i, j] * M[j, ]
  }
  M[, (n + 1):(2 * n)]                          # [I | A^{-1}]
}
adjunta <- function(A) {                        # transposta da matriz de cofatores
  n <- nrow(A); C <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) C[i, j] <- (-1)^(i + j) * det(A[-i, -j, drop = FALSE])
  t(C)
}
A <- matrix(c(2, 1, 1,
              1, 3, 2,
              1, 0, 0), 3, byrow = TRUE)
Ainv <- gauss_jordan_inv(A)
round(Ainv, 4)
max(abs(Ainv - solve(A)))
max(abs(adjunta(A) / det(A) - Ainv))            # A^{-1} = adj(A) / det(A)
round(A %*% adjunta(A), 10)                     # = det(A) I

# Regra de Cramer para Ax = b
b <- c(4, 5, 6)
x_cramer <- sapply(1:3, function(j) { Aj <- A; Aj[, j] <- b; det(Aj) / det(A) })
rbind(cramer = x_cramer, solve = solve(A, b))
