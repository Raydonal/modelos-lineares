# Cap. 6 -- Autovalores: traço, determinante, AB e BA, diagonalização
A <- matrix(c(2, 1, 0,
              1, 3, 1,
              0, 1, 4), 3, byrow = TRUE)
e <- eigen(A)
e$values
c(soma = sum(e$values), traco = sum(diag(A)), produto = prod(e$values), det = det(A))

# Rotação de 90 graus: autovalores complexos +i e -i
eigen(matrix(c(0, -1, 1, 0), 2))$values

# AB e BA têm os mesmos autovalores não nulos
source(file.path("R", "comum", "gerador.R")); g <- gerador(6)
B <- matriz_normal(g, 3, 5); C <- matriz_normal(g, 5, 3)
round(eigen(B %*% C)$values, 6)
round(eigen(C %*% B)$values, 6)

# Diagonalização A = S Lambda S^{-1} e uma matriz não diagonalizável
M <- matrix(c(0.9, 0.2, 0.1, 0.7), 2, byrow = TRUE)
em <- eigen(M); S <- em$vectors
max(abs(S %*% diag(em$values) %*% solve(S) - M))
J <- matrix(c(1, 1, 0, 1), 2, byrow = TRUE)               # bloco de Jordan
ej <- eigen(J); ej$values; round(ej$vectors, 8)           # dois autovetores (quase) paralelos
qr(ej$vectors, tol = 1e-6)$rank

# Idempotentes têm autovalores 0 e 1
P <- matrix(c(2, -2, -4, -1, 3, 4, 1, -2, -3), 3, byrow = TRUE)   # Exercício do Cap. 5
round(eigen(P)$values, 10)
