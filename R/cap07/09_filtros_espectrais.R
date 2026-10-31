# Cap. 7: Pseudo-inversa, tolerância e filtros espectrais num classificador de dígitos
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- cbind(1, as.matrix(dig[, -1]) / 16); y <- dig$digito  # pixels em [0, 1] e intercepto
tr <- 1:1000; te <- 1001:nrow(X)
Y <- outer(y[tr], 0:9, "==") * 1                           # respostas indicadoras (um contra todos)
s <- svd(X[tr, ])
signif(s$d[c(1, 10, 30, 50, 60, 62, 63, 64, 65)], 3)
c(posto_numerico = sum(s$d > 1e-10 * s$d[1]), tol_ginv = sqrt(.Machine$double.eps) * s$d[1])
acuracia <- function(W) mean(max.col(X[te, ] %*% W) - 1 == y[te])
filtro <- function(f) s$v %*% (diag(f) %*% t(s$u) %*% Y)    # W = V diag(f) U' Y
# SVD truncada: f_j = 1/sigma_j para j <= k, 0 depois
tsvd <- sapply(c(5, 10, 20, 30, 40, 50, 62), function(k) {
  W <- filtro(c(1 / s$d[1:k], rep(0, 65 - k))); c(k = k, acuracia = acuracia(W), norma_W = norm(W, "F")) })
round(tsvd, 4)
# Sem tolerância: k = 63 inclui um sigma ~ 1e-14 (pixels sempre apagados no treino)
W63 <- filtro(c(1 / s$d[1:63], rep(0, 2)))
c(acuracia = acuracia(W63), norma_W = norm(W63, "F"))
# Ridge: f_j = sigma_j / (sigma_j^2 + lambda); graus de liberdade efetivos sum sigma^2/(sigma^2+lambda)
ridge <- sapply(10^(-3:2), function(l) {
  W <- filtro(s$d / (s$d^2 + l)); c(lambda = l, gl = sum(s$d^2 / (s$d^2 + l)), acuracia = acuracia(W), norma_W = norm(W, "F")) })
round(ridge, 4)
