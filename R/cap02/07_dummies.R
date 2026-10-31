# Cap. 2 -- A armadilha das variáveis indicadoras: posto incompleto e identificação
dados <- read.csv(file.path("dados", "wage1.csv"))
y <- log(dados$wage)
nordeste <- 1 - dados$northcen - dados$south - dados$west     # categoria omitida nos dados
X <- cbind(intercepto = 1, nordeste, centro_norte = dados$northcen,
           sul = dados$south, oeste = dados$west)
c(colunas = ncol(X), posto = qr(X)$rank)                     # posto incompleto
X %*% c(1, -1, -1, -1, -1) |> abs() |> max()                 # vetor do núcleo: 1 = soma das indicadoras

coef(lm(y ~ X - 1))                                          # R detecta e devolve NA

# Duas parametrizações de posto completo com o mesmo espaço coluna
X1 <- X[, -2]                        # intercepto + 3 regiões (nordeste = referência)
X2 <- X[, -1]                        # 4 indicadoras, sem intercepto (médias por região)
m1 <- lm.fit(X1, y); m2 <- lm.fit(X2, y)
round(coef(m1), 4); round(coef(m2), 4)
max(abs(m1$fitted.values - m2$fitted.values))   # mesmos valores ajustados
tapply(y, max.col(X[, -1]), mean)    # médias por região = coef(m2)
