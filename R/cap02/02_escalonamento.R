# Cap. 2: Forma escalonada reduzida, bases dos quatro subespaços e fatoração de posto
escalonar <- function(A, tol = 1e-10) {
  R <- A; m <- nrow(R); n <- ncol(R); pivos <- integer(0); i <- 1
  for (j in seq_len(n)) {
    if (i > m) break
    k <- which.max(abs(R[i:m, j])) + i - 1      # pivotamento parcial
    if (abs(R[k, j]) < tol) next                # coluna sem pivô
    R[c(i, k), ] <- R[c(k, i), ]                # troca de linhas
    R[i, ] <- R[i, ] / R[i, j]                  # pivô igual a 1
    for (l in setdiff(seq_len(m), i)) R[l, ] <- R[l, ] - R[l, j] * R[i, ]
    pivos <- c(pivos, j); i <- i + 1
  }
  R[abs(R) < tol] <- 0
  list(R = R, pivos = pivos)
}

A <- matrix(c(1, 2, 0, 3,
              2, 4, 1, 7,
              3, 6, 1, 10), nrow = 3, byrow = TRUE)
e <- escalonar(A)
e$R; e$pivos
r <- length(e$pivos)                            # posto = número de pivôs

# Bases: C(A) = colunas pivô de A; R(A) = linhas não nulas de R
base_col <- A[, e$pivos, drop = FALSE]
base_lin <- e$R[1:r, , drop = FALSE]
# N(A): uma solução por variável livre
livres <- setdiff(seq_len(ncol(A)), e$pivos)
base_nul <- sapply(livres, function(f) {
  x <- numeric(ncol(A)); x[f] <- 1; x[e$pivos] <- -e$R[1:r, f]; x
})
base_nul; A %*% base_nul                        # zeros

# Fatoração de posto: A = (colunas pivô) x (linhas não nulas de R)
max(abs(base_col %*% base_lin - A))

# Compatibilidade de Ax = b (Rouché-Capelli)
b1 <- c(1, 3, 4); b2 <- c(1, 3, 5)
c(r_A = r, r_Ab1 = qr(cbind(A, b1))$rank, r_Ab2 = qr(cbind(A, b2))$rank)
