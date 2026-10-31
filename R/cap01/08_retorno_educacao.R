# Cap. 1 -- Retorno da educação: regressão simples como projeção ortogonal
dados <- read.csv(file.path("dados", "wage1.csv"))
y <- log(dados$wage)
x <- dados$educ
n <- length(y)

# Base ortogonal de span{1, x}: u1 = 1 e u2 = x - média(x) (Gram-Schmidt)
u1 <- rep(1, n)
u2 <- x - mean(x)
proj <- function(v, u) sum(v * u) / sum(u * u) * u
y_hat <- proj(y, u1) + proj(y, u2)               # projeção sobre span{1, x}

b1 <- sum(y * u2) / sum(u2 * u2)                 # inclinação
b0 <- mean(y) - b1 * mean(x)                     # intercepto
c(b0 = b0, b1 = b1, retorno_pct = 100 * (exp(b1) - 1))

# Pitágoras e R^2 = cos^2(ângulo entre os vetores centrados)
sq_tot <- sum((y - mean(y))^2); sq_reg <- sum((y_hat - mean(y))^2)
c(R2 = sq_reg / sq_tot, cor2 = cor(x, y)^2)
coef(lm(y ~ x))                                  # conferência
