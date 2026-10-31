# Cap. 5: Matrizes idempotentes: posto = traço, identidade de posto, potências
posto <- function(M) qr(M, tol = 1e-10)$rank
tr <- function(M) sum(diag(M))

# Três idempotentes: oblíqua 2 x 2, centragem e A G (com G inversa generalizada de A)
A1 <- matrix(c(1, 1,
               0, 0), 2, byrow = TRUE)
C5 <- diag(5) - matrix(1 / 5, 5, 5)
A  <- matrix(c(1, 2, 0, 1,  2, 4, 1, 1,  3, 6, 1, 2), 3, byrow = TRUE)
G  <- matrix(0, 4, 3); G[c(1, 3), 1:2] <- solve(A[1:2, c(1, 3)])   # submatriz (Cap. 4)
lista <- list(obliqua = A1, centragem = C5, AG = A %*% G)
t(sapply(lista, function(M) c(erro_idemp = max(abs(M %*% M - M)), simetrica = isSymmetric(M),
                              posto = posto(M), traco = tr(M))))

# Construção de uma idempotente de posto r: A = B (C B)^{-1} C
set.seed(5)
n <- 6; r <- 2
B <- matrix(rnorm(n * r), n); C <- matrix(rnorm(r * n), r)
P <- B %*% solve(C %*% B) %*% C
c(erro_idemp = max(abs(P %*% P - P)), posto = posto(P), traco = tr(P),
  posto_I_menos_P = posto(diag(n) - P))

# Identidade posto(M) + posto(I - M) = n + posto(M - M^2) para M qualquer
M <- P + 0.1 * matrix(rnorm(n * n), n)
c(esquerda = posto(M) + posto(diag(n) - M), direita = n + posto(M - M %*% M))

# Limite de potências de uma matriz de transição: L = 1 pi' é idempotente
Pm <- matrix(c(0.9, 0.1,
               0.3, 0.7), 2, byrow = TRUE)
L <- diag(2); for (k in 1:200) L <- L %*% Pm
L
c(erro_idemp = max(abs(L %*% L - L)), erro_PL = max(abs(Pm %*% L - L)), posto = posto(L))
