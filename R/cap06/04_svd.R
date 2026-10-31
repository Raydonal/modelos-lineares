# Cap. 6 -- Decomposição em valores singulares, subespaços fundamentais e Eckart-Young
A <- matrix(c(1, 2, 0, 1,
              2, 4, 1, 1,
              3, 6, 1, 2), 3, byrow = TRUE)                # a matriz 3 x 4 de posto 2 do Cap. 4
s <- svd(A, nu = 3, nv = 4)
round(s$d, 6)                                               # dois valores singulares não nulos
c(reconstrucao = max(abs(s$u[, 1:3] %*% diag(s$d) %*% t(s$v[, 1:3]) - A)),
  autovalores_AtA = max(abs(sort(eigen(crossprod(A), symmetric = TRUE)$values, TRUE)[1:3] - s$d^2)))
# Bases dos quatro subespaços: C(A), N(A'), espaço linha, N(A)
r <- sum(s$d > 1e-10 * s$d[1])
c(A_v_nulo = max(abs(A %*% s$v[, (r + 1):4])), At_u_nulo = max(abs(t(A) %*% s$u[, (r + 1):3])))

# Eckart-Young: a SVD truncada é a melhor aproximação de posto k
source(file.path("R", "comum", "gerador.R")); g <- gerador(64)
X <- matriz_normal(g, 30, 8)
sx <- svd(X); k <- 2
Xk <- sx$u[, 1:k] %*% diag(sx$d[1:k]) %*% t(sx$v[, 1:k])
c(erro_svd = sum((X - Xk)^2), soma_sigma2_descartados = sum(sx$d[-(1:k)]^2))
# Nenhuma de 2000 aproximações de posto 2 escolhidas ao acaso (projeções sobre planos) faz melhor
outras <- replicate(2000, { U <- qr.Q(qr(matriz_normal(g, 30, k))); sum((X - U %*% t(U) %*% X)^2) })
min(outras)
