# Cap. 8: Influência de cada imagem no primeiro autovalor (derivada de autovalores)
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1]); n <- nrow(X)
S <- cov(X); e <- eigen(S, symmetric = TRUE); l1 <- e$values[1]; q1 <- e$vectors[, 1]
Xc <- sweep(X, 2, colMeans(X)); z <- drop(Xc %*% q1)      # escores na 1a componente
# Sem a observação i: S_(i) = [(n-1) S - n/(n-1) x_i x_i'] / (n-2) (x_i centrado)
# Primeira ordem (d lambda = q' dS q): lambda_(i) - lambda ~ [lambda - n/(n-1) z_i^2] / (n-2)
aprox <- (l1 - n / (n - 1) * z^2) / (n - 2)
exato <- sapply(1:n, function(i) eigen(cov(X[-i, ]), symmetric = TRUE, only.values = TRUE)$values[1] - l1)
c(lambda1 = l1, correlacao = cor(aprox, exato), max_erro_rel = max(abs(aprox - exato)) / max(abs(exato)))
o <- order(exato)[1:5]                                     # remoções que mais reduzem lambda_1
data.frame(imagem = o, digito = dig$digito[o], escore = round(z[o], 1), exato = round(exato[o], 3), aprox = round(aprox[o], 3))
table(digito = dig$digito[order(exato)[1:50]])
