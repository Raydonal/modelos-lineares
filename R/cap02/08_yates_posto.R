# Cap. 2 -- Posto da matriz de delineamento superparametrizada do experimento de aveia
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
ind <- function(f) model.matrix(~ factor(f) - 1)            # indicadoras de um fator
X <- cbind(1, ind(aveia$block), ind(aveia$nitro), ind(aveia$gen))
dim(X)                                                       # 72 x 14
r <- qr(X)$rank
c(posto = r, dim_nucleo = ncol(X) - r)                       # 1 + 5 + 3 + 2 = 11

# Base do núcleo: as três relações "intercepto = soma das indicadoras de cada fator"
N <- cbind(c(1, rep(-1, 6), rep(0, 4), rep(0, 3)),
           c(1, rep(0, 6), rep(-1, 4), rep(0, 3)),
           c(1, rep(0, 6), rep(0, 4), rep(-1, 3)))
max(abs(X %*% N)); qr(N)$rank

# A parametrização padrão do R (contrastes de tratamento) tem posto completo
Xr <- model.matrix(~ factor(block) + factor(nitro) + factor(gen), aveia)
c(colunas = ncol(Xr), posto = qr(Xr)$rank)
y <- aveia$yield
max(abs(fitted(lm(y ~ X - 1)) - fitted(lm(y ~ Xr - 1))))     # mesmo espaço coluna
