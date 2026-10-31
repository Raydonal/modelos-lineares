# Cap. 6: Matrizes definidas positivas: critérios, Cholesky, ordem de Loewner
A <- matrix(c(4, 2, -2,
              2, 10, 4,
              -2, 4, 9), 3, byrow = TRUE)
eigen(A, symmetric = TRUE)$values                          # todos positivos
sapply(1:3, function(k) det(A[1:k, 1:k, drop = FALSE]))   # menores principais líderes (Sylvester)
R <- chol(A); R                                            # A = R'R, R triangular superior
c(erro = max(abs(crossprod(R) - A)), logdet_chol = 2 * sum(log(diag(R))), logdet = log(det(A)))

# Semidefinida: menores líderes não negativos não bastam
B <- matrix(c(0, 0, 0, -1), 2)
c(menor1 = B[1, 1], menor2 = det(B), autovalores = eigen(B)$values)

# Ordem de Loewner: A >= C > 0 implica C^{-1} >= A^{-1}
C <- A - diag(c(1, 2, 3))                                  # A - C é definida positiva
min(eigen(C, symmetric = TRUE)$values)                     # C > 0
min(eigen(solve(C) - solve(A), symmetric = TRUE)$values)   # C^{-1} - A^{-1} >= 0

# Autovalores do suavizador P-spline (Cap. 5) estão em [0, 1]
mc <- read.csv(file.path("dados", "mcycle.csv")); x <- mc$times
h <- (max(x) - min(x)) / 21
Bs <- splines::splineDesign(min(x) + h * (-3:24), x, ord = 4)
D2 <- diff(diag(ncol(Bs)), differences = 2)
S <- Bs %*% solve(crossprod(Bs) + 0.7943282 * crossprod(D2), t(Bs))
ev <- eigen(S, symmetric = TRUE)$values
c(min = min(ev), max = max(ev), iguais_a_1 = sum(ev > 1 - 1e-8), traco = sum(ev))
