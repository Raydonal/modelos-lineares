# Cap. 3 -- Determinantes grandes: LU, log-determinante e estouro numérico
set.seed(1)
n <- 1200
A <- diag(n) * 0.5 + matrix(rnorm(n * n, sd = 0.01), n)  # det ~ 0.5^1200
det(A)                                            # subfluxo: menor que o menor double
determinant(A, logarithm = TRUE)                  # log|det| e sinal, via LU
# log|det| como soma dos logaritmos dos pivôs da LU
lu <- Matrix::expand(Matrix::lu(A))
sum(log(abs(diag(as.matrix(lu$U)))))
n * log(0.5)                                      # referência aproximada
