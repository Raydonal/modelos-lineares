# Cap. 1 -- A média móvel como operador linear: núcleo, imagem e sazonalidade
gas <- read.csv(file.path("dados", "ukgas.csv"))
y <- log(gas$consumo)                        # log torna a sazonalidade aditiva
n <- length(y)                               # 108 trimestres (1960-1986)

# Matriz (n-3) x n da média móvel de 4 termos: (My)_t = (y_t + ... + y_{t+3}) / 4
media_movel <- function(n, k = 4) {
  M <- matrix(0, n - k + 1, n)
  for (t in 1:(n - k + 1)) M[t, t:(t + k - 1)] <- 1 / k
  M
}
M <- media_movel(n)
M[1:3, 1:6]

# Núcleo: padrões de período 4 com soma zero são anulados
s <- rep(c(0.5, -0.2, -0.6, 0.3), length.out = n)  # padrão sazonal (soma 0)
max(abs(M %*% s))                                  # ~ 0
qrM <- qr(M)
c(dim_imagem = qrM$rank, dim_nucleo = n - qrM$rank) # 105 + 3 = 108

# Tendência (média móvel) e componente sazonal estimada
tendencia <- M %*% y
desvio <- y[2:(n - 2)] - tendencia                 # alinhado (aprox.) ao centro da janela
tri <- gas$trimestre[2:(n - 2)]
sazonal <- tapply(desvio, tri, mean)
round(sazonal - mean(sazonal), 3)                  # efeito médio de cada trimestre
