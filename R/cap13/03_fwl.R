# Cap. 13: Frisch-Waugh-Lovell e o viés de variável omitida (wage1)
w <- read.csv(file.path("dados", "wage1.csv")); y <- log(w$wage)
X1 <- cbind(1, exper = w$exper, tenure = w$tenure)  # controles
x2 <- w$educ                                        # de interesse
longo <- lm(y ~ X1 + x2 - 1)
M1 <- function(v) resid(lm(v ~ X1 - 1))             # resíduo em X1
fwl <- lm(M1(y) ~ M1(x2) - 1)               # resíduo em resíduo
round(c(longo = coef(longo)[["x2"]], fwl = coef(fwl)[[1]],
        so_x2_residualizado = coef(lm(y ~ M1(x2) - 1))[[1]]), 6)
max(abs(resid(longo) - resid(fwl)))                 # mesmos resíduos
# Erro-padrão: o do FWL usa gl n - 1 em vez de n - 4
n <- length(y)
round(c(ep_longo = summary(longo)$coef["x2", 2],
        ep_fwl = summary(fwl)$coef[1, 2],
        ep_fwl_corrigido = summary(fwl)$coef[1, 2] *
          sqrt((n - 1) / (n - 4))), 6)
# Viés de omissão: curto = longo + delta * (coef. omitido)
curto <- coef(lm(y ~ x2))[["x2"]]
delta <- coef(lm(X1[, 2:3] ~ x2))[2, ]      # exper, tenure em educ
b <- coef(longo)
round(c(curto = curto, longo = b[["x2"]],
        identidade = b[["x2"]] + sum(delta * b[c("X1exper",
                                                  "X1tenure")])), 6)
round(delta, 4)
