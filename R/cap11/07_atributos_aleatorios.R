# Cap. 11 -- Atributos aleatórios: uma rede de uma camada congelada
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
d <- read.csv(file.path("dados", "digitos.csv"))
X <- cbind(1, as.matrix(d[, -1]) / 16)        # intercepto e pixels
Y <- outer(d$digito, 0:9, "==") * 1           # 10 respostas (one-hot)
tr <- 1:1000; te <- 1001:nrow(d)
g <- gerador(7); M <- 2000
W <- matriz_normal(g, M, 65, sd = 1 / 8)      # pesos congelados
Phi <- pmax(X %*% t(W), 0)                    # ReLU: n x M
acerto <- function(m) {                      # B = A^+ Y (norma mínima)
  A <- Phi[, 1:m, drop = FALSE]; s <- svd(A[tr, ])
  r <- sum(s$d > sqrt(.Machine$double.eps) * s$d[1])  # como ginv
  B <- s$v[, 1:r] %*% (crossprod(s$u[, 1:r], Y[tr, ]) / s$d[1:r])
  prev <- max.col(A[te, ] %*% B, "first") - 1
  c(m = m, posto = r, norma = sqrt(sum(B^2)),
    acerto = mean(prev == d$digito[te]))
}
ms <- c(10, 50, 100, 300, 600, 900, 1000, 1100, 1500, 2000)
round(t(sapply(ms, acerto)), 3)
# Atributos que nunca se acendem no treino (colunas nulas)
sum(colSums(Phi[tr, ]) == 0)
