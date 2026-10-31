# Cap. 4: Inversas reflexivas, de mínimos quadrados, de norma mínima e de Moore-Penrose
A <- matrix(c(1, 2, 0, 1,  2, 4, 1, 1,  3, 6, 1, 2), 3, byrow = TRUE)   # posto 2
ginv_sub <- function(M, i, j) {                 # inversa generalizada por submatriz
  G <- matrix(0, ncol(M), nrow(M)); G[j, i] <- solve(M[i, j]); G
}
G_sub <- ginv_sub(A, 1:2, c(1, 3))
G_mq  <- ginv_sub(crossprod(A), c(1, 3), c(1, 3)) %*% t(A)    # (A'A)^- A'
G_nm  <- t(A) %*% ginv_sub(tcrossprod(A), 1:2, 1:2)           # A'(AA')^-
G_mp  <- MASS::ginv(A)

penrose <- function(G, tol = 1e-10) c(
  "AGA=A"    = max(abs(A %*% G %*% A - A)) < tol,
  "GAG=G"    = max(abs(G %*% A %*% G - G)) < tol,
  "(AG)'=AG" = max(abs(A %*% G - t(A %*% G))) < tol,
  "(GA)'=GA" = max(abs(G %*% A - t(G %*% A))) < tol)
sapply(list(sub = G_sub, mq = G_mq, nm = G_nm, mp = G_mp), penrose)

# Sistema incompatível: G_mq b e G_mp b minimizam ||A x - b||; G_mp b tem a menor norma
b <- c(1, 0, 0)
sol <- list(sub = G_sub %*% b, mq = G_mq %*% b, mp = G_mp %*% b)
sapply(sol, function(x) c(residuo = sqrt(sum((A %*% x - b)^2)), norma = sqrt(sum(x^2))))

# Sistema compatível: G_nm b é a solução de norma mínima (coincide com G_mp b)
b1 <- c(4, 8, 12)
c(dif_nm_mp = max(abs(G_nm %*% b1 - G_mp %*% b1)), norma_mp = sqrt(sum((G_mp %*% b1)^2)),
  norma_sub = sqrt(sum((G_sub %*% b1)^2)))

# Moore-Penrose pela fatoração de posto e como limite da solução ridge
B <- A[, c(1, 3)]; C <- rbind(c(1, 2, 0, 1), c(0, 0, 1, -1))
max(abs(t(C) %*% solve(tcrossprod(C)) %*% solve(crossprod(B)) %*% t(B) - G_mp))
sapply(10^-(1:6), function(l) max(abs(solve(crossprod(A) + l * diag(4), t(A)) - G_mp)))
