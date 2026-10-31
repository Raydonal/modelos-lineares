# Cap. 7: Norma de projeções oblíquas e ângulos principais
norma2 <- function(M) svd(M)$d[1]
# A projeção do Cap. 5: sobre o plano z = 0 ao longo de (1, 1, 1)
P <- matrix(c(1, 0, -1, 0, 1, -1, 0, 0, 0), 3, byrow = TRUE)
c(norma_P = norma2(P), norma_I_menos_P = norma2(diag(3) - P))
# Ângulos principais entre U = C(P) e W^perp = C(P'): cossenos = valores singulares de Qu' Qw
angulos <- function(A, B) { Qa <- qr.Q(qr(A)); Qb <- qr.Q(qr(B)); acos(pmin(1, svd(t(Qa) %*% Qb)$d)) * 180 / pi }
th <- angulos(P[, 1:2], t(P)[, 1:2])
c(angulos_graus = th, um_sobre_cos_max = 1 / cos(max(th) * pi / 180))
# Em dimensão maior: projeção ao longo de um complemento sorteado
source(file.path("R", "comum", "gerador.R")); g <- gerador(72)
B <- matriz_normal(g, 8, 3); C <- matriz_normal(g, 8, 3)
Pq <- B %*% solve(t(C) %*% B) %*% t(C)                    # projeção sobre C(B) ao longo de C(C)^perp
thq <- angulos(B, C)
round(c(norma_P = norma2(Pq), norma_I_menos_P = norma2(diag(8) - Pq), um_sobre_cos_max = 1 / cos(max(thq) * pi / 180)), 6)
round(thq, 2)
