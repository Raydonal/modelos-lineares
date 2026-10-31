# Cap. 9: Imputação pela distribuição condicional (qualidade do ar)
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
aq <- read.csv(file.path("dados", "airquality.csv"))
V <- cbind(lozonio = log(aq$Ozone), rad = aq$Solar.R,
           vento = aq$Wind, temp = aq$Temp)
cc <- complete.cases(V)
c(completas = sum(cc), incompletas = sum(!cc))

# Condicional dos ausentes (m) dados os observados (o) numa linha
condicional <- function(x, mu, S) {
  m <- is.na(x); o <- !m
  B <- S[m, o, drop = FALSE] %*% solve(S[o, o])
  list(m = m, media = drop(mu[m] + B %*% (x[o] - mu[o])),
       cov = S[m, m, drop = FALSE] - B %*% S[o, m, drop = FALSE])
}

# Validação: esconde o ozônio de cada linha completa (sem usá-la)
Vc <- V[cc, ]; z90 <- qnorm(0.95)
val <- t(sapply(seq_len(nrow(Vc)), function(i) {
  mu <- colMeans(Vc[-i, ]); S <- cov(Vc[-i, ])
  x <- Vc[i, ]; x[1] <- NA; ci <- condicional(x, mu, S)
  c(erro = Vc[i, 1] - ci$media, erro_media = Vc[i, 1] - mu[1],
    cobre = abs(Vc[i, 1] - ci$media) < z90 * sqrt(ci$cov))
}))
c(rmse_condicional = sqrt(mean(val[, 1]^2)),
  rmse_media = sqrt(mean(val[, 2]^2)), cobertura90 = mean(val[, 3]))

# Imputação das 42 linhas incompletas: média condicional e sorteio
mu <- colMeans(Vc); S <- cov(Vc); g <- gerador(13)
Vm <- Vs <- V
for (i in which(!cc)) {
  ci <- condicional(V[i, ], mu, S)
  Vm[i, ci$m] <- ci$media
  z <- g$normal(sum(ci$m))
  Vs[i, ci$m] <- ci$media + drop(t(chol(ci$cov)) %*% z)
}
round(head(cbind(media = Vm[!cc, 1], sorteio = Vs[!cc, 1]), 6), 3)
c(var_completas = var(V[cc, 1]), var_media_simples = var(
  ifelse(is.na(V[, 1]), mu[1], V[, 1])),
  var_condicional = var(Vm[, 1]), var_sorteio = var(Vs[, 1]))
