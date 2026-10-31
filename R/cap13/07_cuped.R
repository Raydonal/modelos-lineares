# Cap. 13: Ajuste por covariável em testes A/B (CUPED) e FWL
source(file.path("R", "comum", "gerador.R"))
um <- function(g, n) {                   # um experimento simulado
  x <- g$normal(n)                            # métrica pré-teste
  t <- as.numeric(g$unif(n) < 0.5)            # sorteio do tratamento
  yv <- 1 + 0.1 * t + 0.8 * x + 0.6 * g$normal(n)
  list(x = x, t = t, y = yv) }
g <- gerador(2013); d <- um(g, 500)
dif <- with(d, mean(y[t == 1]) - mean(y[t == 0]))
aj <- coef(lm(y ~ t + x, d))[["t"]]           # regressão com x
th <- with(d, cov(y, x) / var(x))        # CUPED: y - th (x - xbar)
yc <- with(d, y - th * (x - mean(x)))
cup <- mean(yc[d$t == 1]) - mean(yc[d$t == 0])
fwl <- with(d, coef(lm(resid(lm(y ~ x)) ~ resid(lm(t ~ x))))[[2]])
round(c(diferenca = dif, ajustado = aj, cuped = cup, fwl = fwl,
        cor_xy = cor(d$x, d$y)), 4)
# 2000 experimentos: variância de cada estimador do efeito
R <- 2000; est <- matrix(NA, R, 2, dimnames = list(NULL,
                                                 c("dif", "ajust")))
for (r in 1:R) { d <- um(g, 500)
  est[r, ] <- c(mean(d$y[d$t == 1]) - mean(d$y[d$t == 0]),
                coef(lm(y ~ t + x, d))[["t"]]) }
v <- apply(est, 2, var)
round(rbind(media = colMeans(est), var = v), 4)
round(c(razao = v[[2]] / v[[1]],
        teorica = 0.6^2 / (0.8^2 + 0.6^2)), 4)
