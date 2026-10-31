# Cap. 5 -- Variáveis instrumentais como projeção oblíqua (dados de Mroz)
m <- read.csv(file.path("dados", "mroz.csv"))
y <- m$lwage; n <- length(y)
X <- cbind(1, educ = m$educ); Z <- cbind(1, fatheduc = m$fatheduc)

P  <- X %*% solve(crossprod(Z, X)) %*% t(Z)          # projeção sobre C(X) ao longo de C(Z)^perp
H  <- X %*% solve(crossprod(X)) %*% t(X)             # projeção ortogonal sobre C(X)
c(erro_idemp = max(abs(P %*% P - P)), assimetria = max(abs(P - t(P))), traco = sum(diag(P)))
b_mq <- solve(crossprod(X), crossprod(X, y))
b_vi <- solve(crossprod(Z, X), crossprod(Z, y))
cbind(MQ = drop(b_mq), VI = drop(b_vi))
e_vi <- y - P %*% y
c(Z_e = max(abs(crossprod(Z, e_vi))), X_e = max(abs(crossprod(X, e_vi))))

# Norma da projeção oblíqua: 1 / |cos| do ângulo entre educ e fatheduc centrados
r <- cor(m$educ, m$fatheduc)
c(norma_P = svd(P)$d[1], um_sobre_r = 1 / abs(r), r = r)

# Erros-padrão (homocedásticos) de MQ e de VI: a razão é aproximadamente 1/|r|
s_mq <- sqrt(sum((y - H %*% y)^2) / (n - 2)); s_vi <- sqrt(sum(e_vi^2) / (n - 2))
ep_mq <- s_mq * sqrt(solve(crossprod(X))[2, 2])
A <- solve(crossprod(Z, X)); ep_vi <- s_vi * sqrt((A %*% crossprod(Z) %*% t(A))[2, 2])
c(ep_mq = ep_mq, ep_vi = ep_vi, razao = ep_vi / ep_mq)

# Mínimos quadrados em dois estágios com dois instrumentos
Z2 <- cbind(1, m$fatheduc, m$motheduc)
PZ <- Z2 %*% solve(crossprod(Z2), t(Z2))
P2 <- X %*% solve(t(X) %*% PZ %*% X) %*% t(X) %*% PZ
c(erro_idemp = max(abs(P2 %*% P2 - P2)), norma_P2 = svd(P2)$d[1])
drop(solve(t(X) %*% PZ %*% X, t(X) %*% PZ %*% y))
