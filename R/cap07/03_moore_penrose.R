# Cap. 7 -- A inversa de Moore-Penrose pela SVD: propriedades, limite ridge e descontinuidade
A <- matrix(c(1, 2, 0, 1,
              2, 4, 1, 1,
              3, 6, 1, 2), 3, byrow = TRUE)                # posto 2 (Cap. 4)
s <- svd(A); r <- sum(s$d > 1e-10 * s$d[1])
Ap <- s$v[, 1:r] %*% diag(1 / s$d[1:r]) %*% t(s$u[, 1:r])  # V_r Sigma_r^{-1} U_r'
penrose <- c(AGA = max(abs(A %*% Ap %*% A - A)), GAG = max(abs(Ap %*% A %*% Ap - Ap)),
             AG_sim = max(abs(A %*% Ap - t(A %*% Ap))), GA_sim = max(abs(Ap %*% A - t(Ap %*% A))))
signif(c(penrose, igual_ginv = max(abs(Ap - MASS::ginv(A)))), 2)
# (A'A)^+ A' = A^+ e (A')^+ = (A^+)'
mp <- MASS::ginv
signif(c(AtA = max(abs(mp(crossprod(A)) %*% t(A) - Ap)), transp = max(abs(mp(t(A)) - t(Ap)))), 2)
# (BC)^+ difere de C^+ B^+ em geral
B <- matrix(c(1, 1), 1); C <- matrix(c(1, 0), 2)
c(BC_mais = mp(B %*% C), C_mais_B_mais = mp(C) %*% mp(B))
# Limite ridge: (A'A + lambda I)^{-1} A' -> A^+, com erro proporcional a lambda
# erro na norma espectral e o valor exato max_i lambda / (sigma_i (sigma_i^2 + lambda)) do Teorema
erro <- sapply(10^-(1:6), function(l) svd(solve(crossprod(A) + l * diag(4), t(A)) - Ap)$d[1])
exato <- sapply(10^-(1:6), function(l) max(l / (s$d[1:r] * (s$d[1:r]^2 + l))))
rbind(lambda = 10^-(1:6), erro = signif(erro, 4), formula = signif(exato, 4), cota = signif(10^-(1:6) / s$d[r]^3, 4))
# Descontinuidade: A_eps = diag(1, eps) tem A_eps^+ = diag(1, 1/eps), mas A_0^+ = diag(1, 0)
sapply(c(1e-1, 1e-4, 1e-7, 1e-9, 0), function(e) mp(diag(c(1, e)))[2, 2])   # ginv trata 1e-9 como zero
