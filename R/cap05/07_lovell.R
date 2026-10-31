# Cap. 5 -- Ajuste sazonal e regressão: o teorema de Lovell (dados Seatbelts)
s <- read.csv(file.path("dados", "seatbelts.csv")); n <- nrow(s)
y <- log(s$motoristas)
X <- cbind(km = log(s$km), preco = log(s$preco_gasolina), lei = s$lei)
D <- model.matrix(~ factor(s$mes) - 1)                 # indicadoras dos 12 meses
PS <- D %*% solve(crossprod(D), t(D)); MS <- diag(n) - PS
R2_sazonal <- function(v) 1 - sum((MS %*% v)^2) / sum((v - mean(v))^2)
round(c(motoristas = R2_sazonal(y), apply(X, 2, R2_sazonal)), 3)

b_completo <- coef(lm(y ~ X + D - 1))[1:3]             # meses como regressores
b_ambos    <- coef(lm(MS %*% y ~ MS %*% X - 1))        # tudo dessazonalizado
b_so_y     <- coef(lm(MS %*% y ~ X))[-1]                # só a resposta dessazonalizada
b_nenhum   <- coef(lm(y ~ X))[-1]                       # sazonalidade ignorada
tab <- rbind(b_completo, b_ambos, b_so_y, b_nenhum); colnames(tab) <- colnames(X)
round(tab, 4)
e1 <- resid(lm(y ~ X + D - 1)); e2 <- resid(lm(MS %*% y ~ MS %*% X - 1))
max(abs(e1 - e2))                                       # resíduos idênticos
c(s_correto = sqrt(sum(e1^2) / (n - 15)), s_ingenuo = sqrt(sum(e2^2) / (n - 3)))
