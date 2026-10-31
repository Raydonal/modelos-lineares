# Cap. 2 -- Adaptação de posto baixo (LoRA): fatoração de posto e contagem de parâmetros
set.seed(7)
d <- 64; r <- 4
W0 <- matrix(rnorm(d * d), d, d)              # pesos "pré-treinados" (posto completo)
B <- matrix(rnorm(d * r), d, r)               # fatores treináveis
A <- matrix(rnorm(r * d), r, d)
Delta <- B %*% A                              # atualização de posto <= r
c(posto_W0 = qr(W0)$rank, posto_Delta = qr(Delta)$rank,
  posto_W = qr(W0 + Delta)$rank)

# Toda matriz de posto r é um produto B A (fatoração de posto via QR com pivotamento)
f <- qr(Delta, LAPACK = TRUE)
B2 <- qr.Q(f)[, 1:r]                          # base ortonormal de C(Delta)
A2 <- crossprod(B2, Delta)                    # coordenadas: Delta = B2 A2
max(abs(B2 %*% A2 - Delta))

# Parâmetros treináveis: atualização completa (d^2) versus LoRA (2 d r)
dd <- c(768, 4096); rr <- c(4, 8, 16)
tab <- expand.grid(r = rr, d = dd)
tab$completo <- tab$d^2; tab$lora <- 2 * tab$d * tab$r
tab$fracao_pct <- round(100 * tab$lora / tab$completo, 2)
tab
