# Cap. 5: Projeções oblíquas e ortogonais em R^3
norma <- function(v) sqrt(sum(v^2))
# Projeção sobre U = C(B) (o plano z = 0) ao longo de W = ger{w}, w = (1, 1, 1)
B <- cbind(c(1, 0, 0), c(0, 1, 0))
w <- c(1, 1, 1)
C <- MASS::Null(w)                       # colunas: base de W^perp, logo N(C') = W
P <- B %*% solve(t(C) %*% B) %*% t(C)    # P = B (C'B)^{-1} C'
round(P, 4)
c(erro_idemp = max(abs(P %*% P - P)), Pw = norma(P %*% w), simetrica = isSymmetric(P))
y <- c(0, 0, 1)
P %*% y                                  # componente em U de y = (0,0,1)
c(norma_y = norma(y), norma_Py = norma(P %*% y), norma_P = svd(P)$d[1])

# Projeção ortogonal sobre o mesmo plano: simétrica e contração
Po <- B %*% solve(crossprod(B), t(B))
c(norma_Po_y = norma(Po %*% y), norma_Po = svd(Po)$d[1])

# Três fórmulas para a projeção ortogonal sobre C(X), com X de posto incompleto
set.seed(2)
X <- cbind(1, rnorm(8), rnorm(8)); X <- cbind(X, X[, 2] + X[, 3])   # 8 x 4, posto 3
H1 <- X %*% MASS::ginv(crossprod(X)) %*% t(X)
q  <- qr(X); Q <- qr.Q(q)[, 1:q$rank]
H2 <- Q %*% t(Q)
H3 <- X %*% MASS::ginv(X)
c(H1_H2 = max(abs(H1 - H2)), H1_H3 = max(abs(H1 - H3)), traco = sum(diag(H1)))

# Projeção ortogonal no produto interno <u, v>_W = u' W v
W  <- diag(1:8)
PW <- X %*% MASS::ginv(t(X) %*% W %*% X) %*% t(X) %*% W
c(erro_idemp = max(abs(PW %*% PW - PW)), assimetria_PW = max(abs(PW - t(PW))),
  assimetria_WPW = max(abs(W %*% PW - t(W %*% PW))))
