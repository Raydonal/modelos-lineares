# Cap. 1 -- Representação matricial do operador derivada em P_3 e mudança de base
# Base canônica B = (1, t, t^2, t^3): coordenadas de p(t) = c0 + c1 t + c2 t^2 + c3 t^3
D <- matrix(0, 4, 4)
for (k in 1:3) D[k, k + 1] <- k                 # d/dt t^k = k t^(k-1)
D
p <- c(1, -2, 0, 5)                             # p(t) = 1 - 2t + 5t^3
D %*% p                                         # p'(t) = -2 + 15 t^2
D %*% D %*% p                                   # p''(t) = 30 t
D %*% D %*% D %*% D                             # D^4 = 0 (nilpotente)

# Núcleo e imagem: dim N(D) = 1 (constantes), dim Im(D) = 3 (polinômios de grau <= 2)
c(dim_imagem = qr(D)$rank, dim_nucleo = 4 - qr(D)$rank)

# Mudança de base: B' = (1, t-1, (t-1)^2, (t-1)^3); colunas de P = coordenadas de B' em B
P <- cbind(c(1, 0, 0, 0), c(-1, 1, 0, 0), c(1, -2, 1, 0), c(-1, 3, -3, 1))
Pinv <- solve(P)                                # coordenadas de B em B'
Pinv %*% D %*% P                                # matriz de D na base B'
