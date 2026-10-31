# Cap. 2: Posto exato e posto numérico da matriz de imagens de dígitos
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1])                                   # 1797 x 64
nulas <- which(colSums(X != 0) == 0)                         # pixels sempre apagados
colnames(X)[nulas]
f <- qr(X, LAPACK = TRUE)                                    # QR com pivotamento de colunas
d <- abs(diag(qr.R(f)))                                      # |r_kk| decrescentes
signif(d[c(1, 10, 30, 50, 60, 61, 62, 64)], 3)
tau <- c(1e-12, 1e-3, 1e-2, 5e-2)                             # tolerâncias relativas
setNames(sapply(tau, function(t) sum(d > t * d[1])), tau)      # posto numérico
qr(X)$rank                                                   # posto com a tolerância padrão
