# Cap. 9 -- Inicialização de redes: variância das ativações por camada
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
g <- gerador(17); n <- 128; L <- 30; B <- 16
X0 <- matriz_normal(g, n, B)                 # 16 entradas N(0, I)
G <- lapply(1:L, function(l) matriz_normal(g, n, n))
relu <- function(z) pmax(z, 0)
redes <- list(linear_s2_1 = list(f = identity, s2 = 1),
              relu_s2_1 = list(f = relu, s2 = 1),
              relu_s2_2_he = list(f = relu, s2 = 2),
              tanh_s2_1 = list(f = tanh, s2 = 1))
camadas <- c(0, 1, 2, 5, 10, 20, 30)
qs <- sapply(redes, function(rede) {
  X <- X0; qs <- mean(X^2)
  for (l in 1:L) {
    X <- rede$f(sqrt(rede$s2 / n) * G[[l]] %*% X)
    qs <- c(qs, mean(X^2))
  }
  qs
})
q <- qs[camadas + 1, ]; rownames(q) <- camadas
signif(rbind(q, min_1a30 = apply(qs[-1, ], 2, min),
             max_1a30 = apply(qs[-1, ], 2, max)), 3)
signif(mean(X0^2) / 2^camadas, 3)     # previsão: ReLU, s2 = 1
