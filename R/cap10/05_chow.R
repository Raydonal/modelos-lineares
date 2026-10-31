# Cap. 10 -- Teste de Chow: a mesma equação de salários para os sexos?
w <- read.csv(file.path("dados", "wage1.csv"))
y <- log(w$wage); n <- length(y); m <- w$female == 1
X <- cbind(1, w$educ, w$exper, w$tenure); k <- ncol(X)
P <- function(Z) Z %*% solve(crossprod(Z), t(Z))
Xu <- cbind(X * !m, X * m)                    # coeficientes separados
Xd <- cbind(X, m)                             # só o intercepto muda
SQR <- function(Z) drop(t(y) %*% (diag(n) - P(Z)) %*% y)
r <- SQR(X); u <- SQR(Xu); d <- SQR(Xd)
gl_u <- n - 2 * k
F_chow <- ((r - u) / k) / (u / gl_u)          # tudo igual?
F_incl <- ((d - u) / (k - 1)) / (u / gl_u)    # inclinações iguais?
signif(c(SQR_restrito = r, SQR_dummy = d, SQR_separado = u,
        F_chow = F_chow,
        p_chow = pf(F_chow, k, gl_u, lower.tail = FALSE),
        F_incl = F_incl,
        p_incl = pf(F_incl, k - 1, gl_u, lower.tail = FALSE)), 6)
# A hipótese de variâncias iguais, que o teste supõe
s2 <- sapply(c(FALSE, TRUE), function(g)
  sum(resid(lm(y[m == g] ~ X[m == g, -1]))^2) / (sum(m == g) - k))
Fv <- s2[1] / s2[2]
g1 <- sum(!m) - k; g2 <- sum(m) - k
p_v <- 2 * min(pf(Fv, g1, g2), pf(Fv, g1, g2, lower.tail = FALSE))
round(c(s2_homens = s2[1], s2_mulheres = s2[2],
        razao = Fv, p = p_v), 4)
