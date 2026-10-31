# Cap. 13: Mínimos quadrados generalizados: três exemplos pequenos
# (a) Média comum de duas medidas com variâncias 1 e 4
X <- matrix(1, 2, 1); V <- diag(c(1, 4)); Vi <- solve(V)
w <- drop(solve(t(X) %*% Vi %*% X, t(X) %*% Vi))  # pesos de MQG
c(pesos = w, var_mqg = 1 / sum(diag(Vi)),
  var_media = sum(diag(V)) / 4)
# (b) MQG = MQ nos dados "branqueados" por L^{-1}, V = LL'
source(file.path("R", "comum", "gerador.R")); g <- gerador(13)
n <- 6; x <- 1:n; X <- cbind(1, x); y <- 1 + 0.5 * x + g$normal(n)
V <- 0.6^abs(outer(1:n, 1:n, "-"))           # AR(1) com rho = 0,6
L <- t(chol(V))                               # chol devolve L'
b_form <- solve(t(X) %*% solve(V, X), t(X) %*% solve(V, y))
b_branq <- qr.coef(qr(solve(L, X)), solve(L, y))
round(cbind(formula = drop(b_form), branqueado = b_branq), 6)
# (c) Erros equicorrelacionados: com intercepto, MQ = MQG
V <- 0.5 * diag(n) + 0.5                      # rho = 0,5
mqg <- function(X) solve(t(X) %*% solve(V, X), t(X) %*% solve(V, y))
mq <- function(X) solve(crossprod(X), crossprod(X, y))
c(com_intercepto = max(abs(mqg(X) - mq(X))),
  sem_intercepto = max(abs(mqg(X[, 2, drop = FALSE]) -
                           mq(X[, 2, drop = FALSE]))))
# Eficiência de MQ para a média sob AR(1), e a cota de Kantorovich
ef <- function(n, rho) { V <- rho^abs(outer(1:n, 1:n, "-"))
  u <- rep(1, n); l <- eigen(V, symmetric = TRUE)$values
  c(eficiencia = (n^2 / sum(V)) / sum(solve(V, u)),
    kantorovich = 4 * max(l) * min(l) / (max(l) + min(l))^2) }
round(rbind("n=50, rho=0,5" = ef(50, .5),
            "n=50, rho=0,9" = ef(50, .9)), 4)
