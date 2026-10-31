# Cap. 4 -- Cadeia de Markov do mercado de trabalho: sistemas singulares compatíveis
estados <- c("E", "D", "F")       # empregado, desempregado, fora da força de trabalho
P <- matrix(c(0.960, 0.015, 0.025,
              0.250, 0.600, 0.150,
              0.040, 0.030, 0.930), 3, byrow = TRUE, dimnames = list(estados, estados))
r <- c(E = 3000, D = 600, F = 300)                  # renda mensal em cada estado (R$)
A <- diag(3) - P
c(posto = qr(A)$rank, soma_linhas = max(abs(A %*% rep(1, 3))))   # A 1 = 0

# Distribuição estacionária: pi'(I - P) = 0 com 1'pi = 1 (sistema 4 x 3 compatível)
M <- rbind(t(A), 1); d <- c(0, 0, 0, 1)
Gm <- MASS::ginv(M)
data.frame(compativel = max(abs(M %*% Gm %*% d - d)) < 1e-12, posto_M = qr(M)$rank)
pi_est <- drop(Gm %*% d); names(pi_est) <- estados
round(pi_est, 4)
c(taxa_desemprego = unname(pi_est["D"] / (pi_est["E"] + pi_est["D"])),
  participacao = unname(pi_est["E"] + pi_est["D"]))

# Equação de Poisson (I - P) h = r - g 1: compatível se, e só se, g = pi' r
G <- MASS::ginv(A)
compat <- function(g) max(abs(A %*% G %*% (r - g) - (r - g))) < 1e-9
g <- sum(pi_est * r)
data.frame(g = g, compativel_g = compat(g), compativel_2500 = compat(2500))

# Solução geral h = G (r - g 1) + (I - G A) z: muda só por constantes
set.seed(3)
H <- sapply(1:3, function(i) drop(G %*% (r - g) + (diag(3) - G %*% A) %*% rnorm(3, sd = 1e4)))
rownames(H) <- estados; round(H, 1)
round(rbind(E_menos_D = H["E", ] - H["D", ], D_menos_F = H["D", ] - H["F", ]), 2)

# Conferência: h_E - h_D = soma_t (e_E - e_D)' P^t (r - g 1), a vantagem acumulada de renda
v <- c(1, -1, 0); s <- 0
for (t in 0:2000) { s <- s + sum(v * r); v <- drop(v %*% P) }
s
