# Cap. 3: Complemento de Schur, inversa particionada e identidade de Woodbury
set.seed(2)
M <- crossprod(matrix(rnorm(36), 6)) + diag(6)       # 6 x 6 simétrica não singular
i1 <- 1:2; i2 <- 3:6
A <- M[i1, i1]; B <- M[i1, i2]; C <- M[i2, i1]; D <- M[i2, i2]
S <- D - C %*% solve(A, B)                           # complemento de Schur de A
c(det_M = det(M), det_A_det_S = det(A) * det(S))

Ai <- solve(A); Si <- solve(S)
Minv <- rbind(cbind(Ai + Ai %*% B %*% Si %*% C %*% Ai, -Ai %*% B %*% Si),
              cbind(-Si %*% C %*% Ai, Si))
max(abs(Minv - solve(M)))                             # inversa particionada

# Woodbury: (A0 + U W V')^{-1} com A0 diagonal 2000 x 2000 e atualização de posto 5
n <- 2000; k <- 5
a0 <- runif(n, 1, 2)
U <- matrix(rnorm(n * k), n); W <- diag(k); V <- U
woodbury <- function(b) {
  Ab <- b / a0; AU <- U / a0
  Ab - AU %*% solve(solve(W) + crossprod(V, AU), crossprod(V, Ab))
}
b <- rnorm(n)
t_direto <- system.time(x1 <- solve(diag(a0) + U %*% W %*% t(V), b))[["elapsed"]]
t_wood   <- system.time(x2 <- woodbury(b))[["elapsed"]]
c(diferenca = max(abs(x1 - x2)), seg_direto = t_direto, seg_woodbury = t_wood)

# Lema do determinante: det(A0 + U V') = det(A0) det(I + V' A0^{-1} U)
c(direto = determinant(diag(a0) + U %*% t(V))$modulus,
  lema = sum(log(a0)) + determinant(diag(k) + crossprod(V, U / a0))$modulus)
