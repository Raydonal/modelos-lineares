# Cap. 7: Robustez adversarial de um classificador linear
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- cbind(1, as.matrix(dig[, -1]) / 16); y <- dig$digito
tr <- 1:1000; te <- 1001:nrow(X)
Y <- outer(y[tr], 0:9, "==") * 1
s <- svd(X[tr, ])
filtro <- function(f) s$v %*% (diag(f) %*% t(s$u) %*% Y)
corte <- function(k) c(1 / s$d[1:k], rep(0, 65 - k))
modelos <- list(pseudo_inversa = filtro(corte(62)),
                svd_k30 = filtro(corte(30)),
                ridge_10 = filtro(s$d / (s$d^2 + 10)))

# Margens e raios certificados contra cada classe rival j
analisar <- function(W) {
  S <- X[te, ] %*% W; cl <- max.col(S); P <- W[-1, ]  # sem intercepto
  R1 <- R2 <- matrix(Inf, length(te), 10)
  for (j in 1:10) {
    D <- t(P[, cl]) - matrix(P[, j], length(te), 64, byrow = TRUE)
    M <- S[cbind(seq_along(te), cl)] - S[, j]
    rival <- cl != j
    R1[rival, j] <- M[rival] / rowSums(abs(D))[rival]   # norma dual
    R2[rival, j] <- M[rival] / sqrt(rowSums(D^2))[rival]
  }
  list(cl = cl, R1 = R1, R2 = R2)
}
# Pior perturbação l_inf contra a rival mais próxima
ataque <- function(W, a, eps) {
  P <- W[-1, ]; j <- max.col(-a$R1)
  D <- t(P[, a$cl]) - t(P[, j])
  Xa <- X[te, ]
  Xa[, -1] <- pmin(pmax(Xa[, -1] - eps * sign(D), 0), 1)
  mean(max.col(Xa %*% W) - 1 == y[te])
}
t(sapply(modelos, function(W) {
  a <- analisar(W); r_inf <- apply(a$R1, 1, min)
  r_2 <- apply(a$R2, 1, min); ok <- a$cl - 1 == y[te]
  n2 <- norm(W[-1, ], "2")                       # norma espectral
  m <- t(apply(X[te, ] %*% W, 1, sort, decreasing = TRUE))
  cota <- (m[, 1] - m[, 2]) / (sqrt(2) * n2)     # Lipschitz
  round(c(norma2 = n2, acur = mean(ok),
          raio_inf = median(r_inf[ok]), raio_2 = median(r_2[ok]),
          cota_2 = median(cota[ok]),
          rob_0.05 = mean(ok & r_inf > 0.05),
          rob_0.1 = mean(ok & r_inf > 0.1),
          ataque_0.1 = ataque(W, a, 0.1)), 4)
}))
