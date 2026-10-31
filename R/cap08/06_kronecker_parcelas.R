# Cap. 8 -- Parcelas subdivididas no experimento de aveia: a covariância como produto de Kronecker
a <- read.csv(file.path("dados", "yates_oats.csv"))
a <- a[order(a$block, a$gen, a$nitro), ]                   # 18 parcelas (bloco x variedade), 4 subparcelas cada
y <- a$yield; n <- length(y)
X <- model.matrix(~ factor(block) + gen + factor(nitro), a); p <- ncol(X)
# V = I_18 kron (s2 I_4 + s2p J_4): erro de subparcela s2 e de parcela s2p
Vk <- function(s2, s2p) kronecker(diag(18), s2 * diag(4) + s2p * matrix(1, 4, 4))
# Inversa e determinante pelas fórmulas de Kronecker e espectrais
inv_bloco <- function(s2, s2p) (diag(4) - 0.25) / s2 + 0.25 / (s2 + 4 * s2p)   # (s2 I + s2p J)^{-1}
loglik <- function(s2, s2p) {
  Vi <- kronecker(diag(18), inv_bloco(s2, s2p))
  logdet <- 18 * (3 * log(s2) + log(s2 + 4 * s2p))           # autovalores: s2 (54 vezes) e s2 + 4 s2p (18)
  b <- solve(t(X) %*% Vi %*% X, t(X) %*% Vi %*% y); r <- y - X %*% b
  -0.5 * (n * log(2 * pi) + logdet + drop(t(r) %*% Vi %*% r))
}
V0 <- Vk(150, 200)
c(inversa = max(abs(solve(V0) - kronecker(diag(18), inv_bloco(150, 200)))),
  logdet = determinant(V0)$modulus[1] - 18 * (3 * log(150) + log(150 + 800)))
# Máxima verossimilhança (perfilada em beta) para (s2, s2p)
opt <- optim(c(log(150), log(200)), function(t) -loglik(exp(t[1]), exp(t[2])), control = list(reltol = 1e-14))
est <- exp(opt$par); names(est) <- c("s2_subparcela", "s2_parcela"); round(est, 1)
# Erros-padrão por mínimos quadrados generalizados: variedade (entre parcelas) x nitrogênio (dentro)
Vi <- kronecker(diag(18), inv_bloco(est[1], est[2])); covb <- solve(t(X) %*% Vi %*% X)
covols <- sum(lm.fit(X, y)$residuals^2) / (n - p) * solve(crossprod(X))
ep <- function(C) sqrt(diag(C))[c("genMarvellous", "genVictory", "factor(nitro)0.6")]
round(rbind(MQG = ep(covb), MQO_ingenuo = ep(covols)), 2)
