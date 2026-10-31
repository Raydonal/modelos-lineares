# Cap. 13 -- Erros AR(1) na regressão de acidentes (Seatbelts)
s <- read.csv(file.path("dados", "seatbelts.csv")); n <- nrow(s)
y <- log(s$motoristas)
X <- cbind(km = log(s$km), preco = log(s$preco_gasolina),
           lei = s$lei, model.matrix(~ factor(s$mes) - 1))
p <- ncol(X); A <- solve(crossprod(X))
b <- drop(A %*% crossprod(X, y)); e <- y - X %*% b
c(r1 = sum(e[-1] * e[-n]) / sum(e^2))         # autocorrelação lag 1
# Prais-Winsten: L^{-1} para V_ij = rho^|i-j| / (1 - rho^2)
pw <- function(M, r) { M <- as.matrix(M)
  rbind(sqrt(1 - r^2) * M[1, ], M[-1, , drop = FALSE] -
          r * M[-nrow(M), , drop = FALSE]) }
perfil <- function(r) {                       # log-veross. perfilada
  z <- pw(y, r); W <- pw(X, r); bz <- qr.coef(qr(W), z)
  -n / 2 * log(sum((z - W %*% bz)^2) / n) + log(1 - r^2) / 2 }
grade <- seq(0, 0.95, by = 0.001); ll <- sapply(grade, perfil)
rho <- grade[which.max(ll)]
z <- pw(y, rho); W <- pw(X, rho); Ag <- solve(crossprod(W))
bg <- drop(Ag %*% crossprod(W, z))
s2g <- sum((z - W %*% bg)^2) / (n - p)
# Erros-padrão do coeficiente da lei
V <- rho^abs(outer(1:n, 1:n, "-")) / (1 - rho^2)
s2 <- sum(e^2) / (n - p)
s2V <- sum(e^2) / sum(diag((diag(n) - X %*% A %*% t(X)) %*% V))
ep_sand <- sqrt(diag(s2V * A %*% t(X) %*% V %*% X %*% A))
j <- 3
round(c(rho = rho, mq = b[[j]], ep_ingenuo = sqrt(s2 * A[j, j]),
        ep_ar1 = ep_sand[[j]], mqg = bg[[j]],
        ep_mqg = sqrt(s2g * Ag[j, j])), 4)
round(100 * (exp(c(mq = b[[j]], mqg = bg[[j]])) - 1), 1)  # em %
