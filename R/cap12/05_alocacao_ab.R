# Cap. 12 -- Teste A/B/n: quantas unidades dar ao controle?
k <- 4; N <- 600                              # 4 variantes + controle
v <- function(n0) { nt <- (N - n0) / k   # Var(ybar_t - ybar_0)/s2
  c(n0 = n0, nt = nt, var = 1 / nt + 1 / n0,
    cor = (1 / n0) / (1 / nt + 1 / n0)) }
n0_otimo <- N * sqrt(k) / (k + sqrt(k))
round(rbind(igual = v(N / (k + 1)), otima = v(n0_otimo)), 4)
round(1 - v(n0_otimo)["var"] / v(N / (k + 1))["var"], 3)
# Simulação com o gerador portátil: efeitos 0 no controle e
# (0, 0,1, 0,2, 0,3) nas variantes, sigma = 1, 3000 réplicas
source(file.path("R", "comum", "gerador.R"))
g <- gerador(2026); R <- 3000; ef <- c(0, 0, .1, .2, .3)
sim <- function(n0) { nt <- (N - n0) / k
  grupo <- rep(1:5, c(n0, rep(nt, k)))
  t(replicate(R, {
    ybar <- tapply(ef[grupo] + g$normal(N), grupo, mean)
    ybar[2:5] - ybar[1] })) }
for (n0 in c(N / (k + 1), n0_otimo)) {
  d <- sim(n0)
  print(round(c(n0 = n0, media_d4 = mean(d[, 4]),
    var_media = mean(apply(d, 2, var)),
    cor_12 = cor(d[, 1], d[, 2])), 4))
}
