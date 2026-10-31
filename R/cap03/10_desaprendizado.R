# Cap. 3 -- Desaprendizado de máquina: remover registros por Woodbury
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
db <- read.csv(file.path("dados", "diabetes.csv"))
X <- cbind(1, as.matrix(db[, 1:10])); y <- db$y
n <- nrow(X); p <- ncol(X)

# O modelo guardado: X'X, sua inversa G, o estimador e os resíduos
A <- crossprod(X); G <- solve(A); beta <- G %*% crossprod(X, y)
e <- drop(y - X %*% beta); h <- rowSums((X %*% G) * X)

# Remoção em bloco: beta_(S) = beta - G X_S' (I - H_SS)^{-1} e_S
esquecer <- function(S) {
  XS <- X[S, , drop = FALSE]
  K <- diag(length(S)) - XS %*% G %*% t(XS)  # I - H_SS
  beta - G %*% t(XS) %*% solve(K, e[S])
}
retreinar <- function(S) qr.coef(qr(X[-S, ]), y[-S])

g <- gerador(3)
S_acaso <- sort(order(g$unif(n))[1:20])     # 20 pedidos ao acaso
S_alav <- order(h, decreasing = TRUE)[1:20] # 20 de maior alavanca
for (S in list(acaso = S_acaso, alavanca = S_alav)) {
  bS <- esquecer(S)
  print(c(dif_retreino = max(abs(bS - retreinar(S))),
          h_medio = mean(h[S]),
          mudanca_ajuste = sqrt(mean((X %*% (beta - bS))^2)),
          resid_antes = mean(abs(e[S])),
          resid_depois = mean(abs(y[S] - X[S, ] %*% bS))))
}

# Pedidos um a um: 20 atualizações de posto um (Sherman-Morrison)
Gs <- G; bs <- crossprod(X, y)
for (i in S_acaso) {
  x <- X[i, ]; Gx <- Gs %*% x
  Gs <- Gs + Gx %*% t(Gx) / as.numeric(1 - crossprod(x, Gx))
  bs <- bs - x * y[i]
}
G_ret <- solve(crossprod(X[-S_acaso, ]))
c(dif_inversa = max(abs(Gs - G_ret)) / max(abs(Gs)),
  dif_beta = max(abs(Gs %*% bs - retreinar(S_acaso))))

# O vazamento: d = A (beta - beta_(i)) = x_i e_(i)
i <- S_acaso[1]; bi <- esquecer(i)
d <- drop(A %*% (beta - bi))
x_rec <- d / d[1]                      # o intercepto fixa a escala
y_rec <- sum(x_rec * bi) + d[1]        # y_i = x_i' beta_(i) + e_(i)
c(paciente = i, erro_x = max(abs(x_rec - X[i, ])),
  erro_y = abs(y_rec - y[i]))

# Custo num problema grande (n = 200 000, p = 50, k = 20)
set.seed(1); N <- 2e5
Xg <- cbind(1, matrix(rnorm(N * 49), N))
yg <- drop(Xg %*% rnorm(50) + rnorm(N))
Gg <- solve(crossprod(Xg)); bg <- Gg %*% crossprod(Xg, yg)
XS <- Xg[1:20, ]
t_ret <- system.time(qr.coef(qr(Xg[-(1:20), ]), yg[-(1:20)]))[[3]]
t_upd <- system.time(for (k in 1:100) {
  K <- diag(20) - XS %*% Gg %*% t(XS)
  bg - Gg %*% t(XS) %*% solve(K, yg[1:20] - XS %*% bg)
})[[3]] / 100
c(retreinar_s = t_ret, atualizar_s = t_upd, razao = t_ret / t_upd)
