# Cap. 9: Anomalias pela distância de Mahalanobis (dígitos)
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1]) / 16; y <- dig$digito
tr <- 1:1000; te <- 1001:nrow(X)
Z <- X[tr, ][y[tr] == 1, ]                   # o "normal": dígitos 1
mu <- colMeans(Z); S <- cov(Z)
c(n = nrow(Z), posto = qr(S)$rank,
  constantes = sum(apply(Z, 2, sd) == 0))

auc <- function(s, anom) {                   # Mann-Whitney
  r <- rank(s); n1 <- sum(anom); n0 <- sum(!anom)
  (sum(r[anom]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
D <- sweep(X[te, ], 2, mu)
anom <- y[te] != 1; oito <- y[te] %in% c(1, 8)
d2 <- function(M, Si) rowSums((M %*% Si) * M)
eu <- d2(D, diag(64))
res <- rbind(euclidiana = c(lambda = NA, auc = auc(eu, anom),
                           auc_8 = auc(eu[oito], anom[oito])))
for (l in c(0.001, 0.01, 0.1, 1)) {
  Si <- solve(S + l * diag(64)); m <- d2(D, Si)
  res <- rbind(res, mahalanobis = c(l, auc(m, anom),
                                    auc(m[oito], anom[oito])))
}
round(res, 4)

# Calibração com lambda = 0,01: o limiar qui-quadrado não vale
Si <- solve(S + 0.01 * diag(64))
dt <- d2(sweep(Z, 2, mu), Si); d1 <- d2(D[!anom, ], Si)
c(media_treino = mean(dt), media_teste_1 = mean(d1),
  falsos_pos_qui2 = mean(d1 > qchisq(0.99, 64)),
  falsos_pos_q99_treino = mean(d1 > quantile(dt, 0.99)))
