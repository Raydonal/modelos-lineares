# Cap. 6: Raio espectral, série de Neumann e matrizes estocásticas positivas
raio <- function(M) max(Mod(eigen(M)$values))
# Uma matriz com raio espectral < 1 mas norma "grande": A^k -> 0 depois de crescer
A <- matrix(c(0.5, 4, 0, 0.6), 2, byrow = TRUE)
c(raio = raio(A), max_entrada = max(abs(A)))
pot <- function(M, k) { R <- diag(nrow(M)); for (i in seq_len(k)) R <- R %*% M; R }
sapply(c(1, 2, 5, 10, 20, 50), function(k) max(abs(pot(A, k))))

# Série de Neumann (insumo-produto de Leontief, coeficientes técnicos hipotéticos)
Tc <- matrix(c(0.20, 0.30, 0.10,
               0.10, 0.25, 0.20,
               0.15, 0.10, 0.30), 3, byrow = TRUE,
             dimnames = list(c("agro", "indústria", "serviços"), c("agro", "indústria", "serviços")))
raio(Tc)
Lt <- solve(diag(3) - Tc)                                   # matriz inversa de Leontief
soma <- diag(3); termo <- diag(3)
for (k in 1:60) { termo <- termo %*% Tc; soma <- soma + termo }
c(erro_neumann = max(abs(soma - Lt)))
round(Lt, 4)
round(colSums(Lt), 4)                                       # multiplicadores de produção

# Contração da oscilação por uma matriz estocástica positiva
P <- matrix(c(0.960, 0.015, 0.025,
              0.250, 0.600, 0.150,
              0.040, 0.030, 0.930), 3, byrow = TRUE)        # mercado de trabalho (Cap. 4)
delta <- min(P); c(delta = delta, fator = 1 - 3 * delta)
osc <- function(v) max(v) - min(v)
v <- c(1, 0, 0); r <- numeric(6)
for (k in 1:6) { w <- drop(P %*% v); r[k] <- osc(w) / osc(v); v <- w }
round(r, 4)                                                  # sempre <= 1 - n delta
