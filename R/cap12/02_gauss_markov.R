# Cap. 12 -- Gauss-Markov: três estimadores não viesados da inclinação
source(file.path("R", "comum", "gerador.R"))
x <- 0:4; X <- cbind(1, x); n <- 5
A <- rbind(mq = solve(crossprod(X), t(X))[2, ],  # a'y para cada um
           pontas = c(-1, 0, 0, 0, 1) / 4,
           pares = c(-1, -1, 0, 1, 1) / 6)
round(A, 3)
M <- cbind(A %*% X, rowSums(A^2))            # a'X = (0, 1)?
colnames(M) <- c("a'1", "a'x", "var/s2"); round(M, 4)
# Monte Carlo com o gerador portátil: beta = (2, 0,5), sigma = 1
g <- gerador(12); R <- 20000
E <- matriz_normal(g, R, n)
Y <- matrix(2 + 0.5 * x, R, n, byrow = TRUE) + E
est <- Y %*% t(A)
round(rbind(media = colMeans(est), var = apply(est, 2, var)), 4)
# Versão matricial: Cov(Ay) - Cov(b_mq) é semidefinida positiva
Ap <- rbind(c(1, 0, 0, 0, 0), A["pontas", ])    # (y_1, inclinação)
Cmq <- solve(crossprod(X)); Cp <- tcrossprod(Ap)
round(eigen(Cp - Cmq)$values, 4)
