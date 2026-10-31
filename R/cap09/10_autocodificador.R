# Cap. 9 -- Autocodificador linear, PCA probabilística e VAE linear
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1]) / 16
tr <- X[1:1000, ]; te <- X[-(1:1000), ]
n <- nrow(tr); p <- ncol(tr)
mu <- colMeans(tr); Rtr <- sweep(tr, 2, mu); Rte <- sweep(te, 2, mu)
S <- crossprod(Rtr) / n                      # EMV
e <- eigen(S, symmetric = TRUE)
d <- pmax(e$values, 0); Q <- e$vectors       # zeros numéricos

# EMV de Tipping e Bishop (s2 livre) ou W ótimo com s2 fixo
ppca <- function(k, s2 = mean(d[-(1:k)])) {
  list(W = Q[, 1:k, drop = FALSE] %*%
         diag(sqrt(pmax(d[1:k] - s2, 0)), k), s2 = s2)
}
# Log-densidade média de N(mu, WW' + s2 I), por Woodbury
logver <- function(m, R) {
  k <- ncol(m$W); G <- diag(k) + crossprod(m$W) / m$s2
  U <- R %*% m$W
  quad <- (rowSums(R^2) - rowSums((U %*% solve(G)) * U) / m$s2) / m$s2
  logdet <- p * log(m$s2) + determinant(G)$modulus
  mean(-0.5 * (p * log(2 * pi) + logdet + quad))
}
tab <- t(sapply(c(1, 2, 5, 10, 20, 30, 40, 50, 55), function(k) {
  P <- tcrossprod(Q[, 1:k]); m <- ppca(k)
  c(k = k, erro_pca_teste = mean(rowSums((Rte - Rte %*% P)^2)),
    sigma2 = m$s2, logver_treino = logver(m, Rtr),
    logver_teste = logver(m, Rte))
}))
print(signif(tab, 4))

# ELBO = log p(x) com o codificador exato; folga = KL(q || post.)
kl_normal <- function(m1, C1, m2, C2) {
  C2i <- solve(C2); dm <- m2 - m1
  0.5 * (sum(diag(C2i %*% C1)) + sum(dm * (C2i %*% dm)) - length(m1) +
         determinant(C2)$modulus - determinant(C1)$modulus)
}
elbo <- function(r, m, mq, Cq) {
  res <- r - m$W %*% mq; k <- ncol(m$W)
  rec <- -0.5 * (p * log(2 * pi * m$s2) + (sum(res^2) +
    sum(diag(m$W %*% Cq %*% t(m$W)))) / m$s2)
  rec - kl_normal(mq, Cq, rep(0, k), diag(k))
}
m <- ppca(10); r <- Rte[1, ]; G <- diag(10) + crossprod(m$W) / m$s2
m_post <- solve(G, crossprod(m$W, r)) / m$s2; C_post <- solve(G)
m_pca <- qr.solve(m$W, r)                    # outro codificador
lp <- logver(m, t(r))
round(c(log_p = lp, elbo_exata = elbo(r, m, m_post, C_post),
        folga_pseudo_inversa = lp - elbo(r, m, m_pca, C_post),
        kl = kl_normal(m_pca, C_post, m_post, C_post)), 6)

# Colapso posterior: k = 20 dimensões latentes, sigma2 fixo
t(sapply(c(0.002, 0.01, 0.05, 0.1, 0.3), function(s2) {
  m <- ppca(20, s2)
  c(sigma2 = s2, ativas = sum(sqrt(colSums(m$W^2)) > 0),
    logver_teste = round(logver(m, Rte), 2))
}))
round(d[c(1, 5, 10, 20, 30, 40)], 4)
