# Cap. 4: Compatibilidade e solução geral de A x = b (e de A X C = B)
A <- matrix(c(1, 2, 0, 1,  2, 4, 1, 1,  3, 6, 1, 2), 3, byrow = TRUE)   # posto 2
G <- MASS::ginv(A)
compativel <- function(b) isTRUE(all.equal(drop(A %*% G %*% b), b))  # teste A G b = b
b1 <- c(4, 8, 12); b2 <- c(1, 0, 0)
c(b1 = compativel(b1), b2 = compativel(b2))

# solução geral x = G b + (I - G A) z
set.seed(1)
x <- function(z) drop(G %*% b1 + (diag(4) - G %*% A) %*% z)
X <- sapply(1:5, function(i) x(rnorm(4)))       # cinco soluções
round(X, 3)
max(abs(A %*% X - b1))                          # todas resolvem o sistema
qr(diag(4) - G %*% A)$rank                      # dim N(A) = 4 - posto(A) = 2

# k'x é o mesmo para todas as soluções se, e somente se, k' está no espaço linha de A
k_linha <- c(3, 6, 1, 2) - c(1, 2, 0, 1)        # combinação de linhas de A
k_fora  <- c(1, 0, 0, 0)
round(rbind(k_linha = drop(k_linha %*% X), k_fora = drop(k_fora %*% X)), 3)

# A X C = B: compatível se A A^- B C^- C = B; solução geral A^- B C^- + Z - A^- A Z C C^-
C <- matrix(c(1, 2, 2, 4), 2)                   # posto 1
B <- A %*% matrix(1:8, 4) %*% C
Gc <- MASS::ginv(C)
max(abs(A %*% G %*% B %*% Gc %*% C - B))
Z <- matrix(rnorm(8), 4)
Xs <- G %*% B %*% Gc + Z - G %*% A %*% Z %*% C %*% Gc
max(abs(A %*% Xs %*% C - B))
