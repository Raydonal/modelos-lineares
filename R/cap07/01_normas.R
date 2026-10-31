# Cap. 7: Normas matriciais: fórmulas das normas induzidas, Frobenius e raio espectral
A <- matrix(c(1, -2, 3,
              0, 4, -1,
              2, 1, 1), 3, byrow = TRUE)
s <- svd(A)$d
c(norma1 = norm(A, "O"), max_soma_colunas = max(colSums(abs(A))),
  norma_inf = norm(A, "I"), max_soma_linhas = max(rowSums(abs(A))),
  norma2 = norm(A, "2"), sigma1 = s[1], frobenius = norm(A, "F"), raiz_soma_sigma2 = sqrt(sum(s^2)))
c(raio_espectral = max(Mod(eigen(A)$values)), norma2_inversa = norm(solve(A), "2"), um_sobre_sigma_min = 1 / min(s))
# O máximo de ||Ax||/||x|| sobre 10 000 vetores ao acaso nunca passa de sigma_1
source(file.path("R", "comum", "gerador.R")); g <- gerador(70)
X <- matriz_normal(g, 3, 10000)
max(sqrt(colSums((A %*% X)^2)) / sqrt(colSums(X^2)))
# Invariância por matrizes ortogonais
Q1 <- qr.Q(qr(matriz_normal(g, 3, 3))); Q2 <- qr.Q(qr(matriz_normal(g, 3, 3)))
c(norma2 = norm(Q1 %*% A %*% Q2, "2"), frobenius = norm(Q1 %*% A %*% Q2, "F"), norma1 = norm(Q1 %*% A %*% Q2, "O"))
