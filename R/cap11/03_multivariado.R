# Cap. 11 -- Modelo multivariado: a forma vetorizada com Kronecker
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
g <- gerador(11); n <- 50
X <- cbind(1, matriz_normal(g, n, 2))         # n x 3
B <- matrix(c(1, 2, -1, 0, 0.5, 3), 3, 2)     # duas respostas
L <- t(chol(matrix(c(1, 0.8, 0.8, 1), 2)))    # Sigma = L L'
E <- matriz_normal(g, n, 2) %*% t(L)          # linhas ~ N(0, Sigma)
Y <- X %*% B + E
B_mq <- solve(crossprod(X), crossprod(X, Y))  # coluna a coluna
Xv <- kronecker(diag(2), X)                   # vec Y = (I(x)X) vec B
S <- crossprod(E) / n                         # qualquer Sigma > 0
W <- kronecker(solve(S), diag(n))             # Cov(vec E)^{-1}
b_g <- solve(t(Xv) %*% W %*% Xv, t(Xv) %*% W %*% c(Y))
c(dif_mqg_mq = max(abs(b_g - c(B_mq))))
round(B_mq, 3)
