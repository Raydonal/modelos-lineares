# Cap. 9: Carteira de variância mínima e valor em risco
eu <- read.csv(file.path("dados", "eustock.csv"))
r <- 100 * diff(log(as.matrix(eu[, -1])))   # retornos diários em %
n <- nrow(r); um <- rep(1, 4)
m <- colMeans(r); S <- cov(r)
round(rbind(media = m, dp = sqrt(diag(S))), 4)
round(cor(r), 3)

# Variância mínima: w = S^{-1} 1 / (1' S^{-1} 1)
Si1 <- solve(S, um); w <- Si1 / sum(Si1)
round(c(w, dp_minvar = sqrt(1 / sum(Si1)),
        dp_iguais = sqrt(drop(t(um / 4) %*% S %*% um / 4))), 4)

# Fronteira: var(alvo) = (a t^2 - 2 b t + c) / (a c - b^2)
a <- sum(Si1); b <- sum(solve(S, m)); c_ <- drop(t(m) %*% solve(S, m))
fronteira <- function(t)
  sqrt((a * t^2 - 2 * b * t + c_) / (a * c_ - b^2))
round(c(retorno_minvar = b / a, dp_alvo_0.1 = fronteira(0.1)), 4)

# Valor em risco a 1% da carteira de variância mínima
rp <- drop(r %*% w)
var_normal <- -(mean(rp) + qnorm(0.01) * sd(rp))
c(VaR_normal = var_normal, VaR_empirico = -unname(quantile(rp, 0.01)),
  excedencias = mean(rp < -var_normal),
  curtose = mean((rp - mean(rp))^4) / var(rp)^2)

# Fora da amostra: estima na 1a metade, avalia na 2a
h <- 1:floor(n / 2)
S1 <- cov(r[h, ]); w1 <- solve(S1, um); w1 <- w1 / sum(w1)
c(dp_minvar_fora = sd(r[-h, ] %*% w1),
  dp_iguais_fora = sd(r[-h, ] %*% um / 4))
