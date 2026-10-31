# Cap. 1: Camadas lineares: composição de funções afins é afim
set.seed(1)
W1 <- matrix(rnorm(4 * 3), 4, 3); b1 <- rnorm(4)   # camada 1: R^3 -> R^4
W2 <- matrix(rnorm(2 * 4), 2, 4); b2 <- rnorm(2)   # camada 2: R^4 -> R^2
x <- c(1, -2, 0.5)

rede <- function(x) W2 %*% (W1 %*% x + b1) + b2    # duas camadas, sem ativação
W <- W2 %*% W1; b <- W2 %*% b1 + b2                # uma única camada equivalente
cbind(rede(x), W %*% x + b)

# Coordenadas homogêneas: a camada afim vira um produto matricial
aumentar <- function(W, b) rbind(cbind(W, b), c(rep(0, ncol(W)), 1))
M <- aumentar(W2, b2) %*% aumentar(W1, b1)
M %*% c(x, 1)                                      # última coordenada = 1

# Com a ativação ReLU a composição deixa de ser linear
relu <- function(z) pmax(z, 0)
rede_relu <- function(x) W2 %*% relu(W1 %*% x + b1) + b2
cbind(rede_relu(2 * x), 2 * rede_relu(x) - b2)     # f(2x) != 2 f(x) - b2 em geral
