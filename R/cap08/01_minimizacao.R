# Cap. 8 -- Minimização de funções quadráticas, com e sem restrições lineares
A <- matrix(c(2, -1, -1, -1, 2, -1, -1, -1, 2), 3)          # Laplaciano do triângulo: posto 2, A 1 = 0
eigen(A, symmetric = TRUE)$values
f <- function(x, b) 0.5 * drop(t(x) %*% A %*% x) - sum(b * x)
b1 <- c(1, -2, 1)                                          # b em C(A) (soma zero): há mínimo
b2 <- c(1, 1, 1)                                           # b fora de C(A): f é ilimitada inferiormente
G <- MASS::ginv(A)
x1 <- G %*% b1
c(compativel = max(abs(A %*% x1 - b1)), minimo = f(x1, b1), formula = -0.5 * drop(t(b1) %*% G %*% b1))
sapply(c(0, 5, -3), function(t) f(x1 + t, b1))              # x1 + t 1 também minimiza (direção de N(A))
sapply(c(1, 10, 100), function(t) f(t * rep(1, 3), b2))     # decresce sem limite

# Com restrição linear C x = d: sistema de Lagrange [[A, C'], [C, 0]]
A2 <- diag(c(1, 2, 3)); bb <- c(1, 1, 1)
C <- matrix(c(1, 1, 1), 1); d <- 1
M <- rbind(cbind(A2, t(C)), cbind(C, 0))
sol <- solve(M, c(bb, d)); x <- sol[1:3]; mult <- sol[4]
c(x = x, multiplicador = mult, restricao = sum(x) - d)
# Conferência: o gradiente A x - b é proporcional a C' (condição de otimalidade)
drop(A2 %*% x - bb) / c(C)
# Minimizar sobre x2 uma forma quadrática particionada dá o complemento de Schur
S <- matrix(c(4, 2, 1, 2, 3, 1, 1, 1, 2), 3); x1v <- c(1, 2)
S11 <- S[1:2, 1:2]; S12 <- S[1:2, 3, drop = FALSE]; S22 <- S[3, 3]
x2_otimo <- -drop(solve(S22, t(S12) %*% x1v))
c(min_numerico = optimize(function(t) drop(t(c(x1v, t)) %*% S %*% c(x1v, t)), c(-10, 10))$objective,
  schur = drop(t(x1v) %*% (S11 - S12 %*% solve(S22) %*% t(S12)) %*% x1v), x2_otimo = x2_otimo)
