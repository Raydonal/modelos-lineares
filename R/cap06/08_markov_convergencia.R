# Cap. 6 -- Velocidade de convergência do mercado de trabalho para o equilíbrio
estados <- c("E", "D", "F")
P <- matrix(c(0.960, 0.015, 0.025,
              0.250, 0.600, 0.150,
              0.040, 0.030, 0.930), 3, byrow = TRUE, dimnames = list(estados, estados))
e <- eigen(P)
e$values                                         # 1 e dois autovalores reais menores que 1
pi <- Re(eigen(t(P))$vectors[, 1]); pi <- pi / sum(pi); names(pi) <- estados
round(pi, 4)
# Decomposição espectral (P é diagonalizável): P^t = 1 pi' + sum lambda_i^t s_i w_i'
S <- e$vectors; W <- solve(S)                    # linhas de W: autovetores à esquerda
dimnames(S) <- list(estados, c("v1", "v2", "v3")); round(S, 3)   # autovetores à direita
P_t <- function(t) Re(S %*% diag(e$values^t) %*% W)
c(erro_t12 = max(abs(P_t(12) - Reduce(`%*%`, rep(list(P), 12)))))
meia_vida <- log(2) / -log(e$values[2])
c(lambda2 = e$values[2], meia_vida_meses = meia_vida)

# Desempregado hoje: distância à distribuição de equilíbrio ao longo do tempo
x <- c(0, 1, 0)
dist <- sapply(c(1, 3, 6, 12, 24, 36), function(t) sum(abs(x %*% P_t(t) - pi)))
round(rbind(meses = c(1, 3, 6, 12, 24, 36), distancia = dist, razao_lambda2 = dist / e$values[2]^c(1, 3, 6, 12, 24, 36)), 4)
# Uma política que dobra a taxa de saída do desemprego para o emprego
P2 <- P; P2["D", ] <- c(0.50, 0.35, 0.15)
e2 <- eigen(P2)$values
pi2 <- Re(eigen(t(P2))$vectors[, 1]); pi2 <- pi2 / sum(pi2)
round(c(lambda2 = e2[2], meia_vida = log(2) / -log(e2[2]), desemprego = pi2[2] / (pi2[1] + pi2[2])), 4)
