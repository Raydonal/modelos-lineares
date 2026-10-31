# Cap. 11: Salários: regressão e média condicional
w <- read.csv(file.path("dados", "wage1.csv"))
y <- log(w$wage); e <- w$educ; n <- length(y)
nj <- table(e); ej <- as.numeric(names(nj))   # 18 caselas
mj <- tapply(y, e, mean)                      # média condicional
D <- outer(e, ej, "==") * 1                   # modelo saturado
X1 <- cbind(1, e)                             # reta
X2 <- cbind(X1, e^2)                          # parábola
X3 <- cbind(X1, pmax(e - 12, 0))              # reta quebrada em 12
P <- function(Z, v = y) Z %*% qr.solve(Z, v)
# MQ com os indivíduos = MQ ponderado nas médias das caselas
b_ind <- qr.solve(X1, y)
b_cel <- lm(mj ~ ej, weights = as.numeric(nj))$coef
c(dif = max(abs(b_ind - b_cel)), b0 = b_ind[[1]], b1 = b_ind[[2]])
# Que fração da variação entre caselas cada forma capta?
entre <- sum((P(D) - mean(y))^2)
capta <- function(Z) sum((P(Z) - mean(y))^2) / entre
round(c(reta = capta(X1), parabola = capta(X2),
        quebrada = capta(X3)), 3)
b3 <- qr.solve(X3, y)
round(c(incl_ate_12 = b3[[2]], incl_depois = b3[[2]] + b3[[3]]), 4)
b2 <- qr.solve(X2, y)
round(b2[2] + 2 * b2[3] * c(8, 12, 16), 4)    # retornos marginais
