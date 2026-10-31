# Cap. 6 -- Componentes principais dos dígitos manuscritos e o teorema de Eckart-Young
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1]); n <- nrow(X)            # 1797 imagens de 8 x 8 pixels
Xc <- scale(X, scale = FALSE)                      # centra (não padroniza)
s <- svd(Xc)
var_exp <- s$d^2 / sum(s$d^2)
round(cumsum(var_exp)[c(1, 2, 5, 10, 20, 30, 40)], 3)
sapply(c(0.80, 0.90, 0.95, 0.99), function(q) which(cumsum(var_exp) >= q)[1])
# Autovalores da covariância amostral = sigma^2 / (n - 1)
c(diferenca = max(abs(eigen(cov(X), symmetric = TRUE)$values - s$d^2 / (n - 1))))
# Erro de reconstrução com k componentes = soma dos sigma^2 descartados (Eckart-Young)
erro <- function(k) { Xk <- s$u[, 1:k] %*% diag(s$d[1:k], k) %*% t(s$v[, 1:k]); sum((Xc - Xk)^2) }
ks <- c(2, 10, 20, 40)
round(rbind(k = ks, erro = sapply(ks, erro), descartados = sapply(ks, function(k) sum(s$d[-(1:k)]^2)),
            relativo = sapply(ks, erro) / sum(s$d^2)), 3)
# Classificação por vizinho mais próximo (1-NN) no espaço das k primeiras componentes
treino <- 1:1000; teste <- 1001:n
acuracia <- function(k) {
  Z <- Xc %*% s$v[, 1:k, drop = FALSE]             # (a SVD usou todas as imagens, sem os rótulos)
  d2 <- outer(rowSums(Z[teste, , drop = FALSE]^2), rowSums(Z[treino, , drop = FALSE]^2), "+") -
    2 * Z[teste, , drop = FALSE] %*% t(Z[treino, , drop = FALSE])
  mean(dig$digito[treino][apply(d2, 1, which.min)] == dig$digito[teste])
}
round(sapply(c(2, 5, 10, 20, 40, 64), acuracia), 4)
