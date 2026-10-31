# Cap. 5: Subespaços encaixados, somas de quadrados e restrições (dados wage1)
w  <- read.csv(file.path("dados", "wage1.csv"))
y  <- log(w$wage); n <- length(y)
proj <- function(X) X %*% solve(crossprod(X), t(X))   # posto coluna completo
sq <- function(v) sum(v^2)
X0 <- matrix(1, n)                                    # só o intercepto
X1 <- cbind(1, w$educ)                                # + escolaridade
X2 <- cbind(X1, w$exper, w$tenure)                    # + experiência e permanência
P0 <- proj(X0); P1 <- proj(X1); P2 <- proj(X2)
c(P2P1_P1 = max(abs(P2 %*% P1 - P1)), P1P2_P1 = max(abs(P1 %*% P2 - P1)),
  P2_menos_P1_idemp = max(abs((P2 - P1) %*% (P2 - P1) - (P2 - P1))))

# Decomposição ortogonal de y - ybar 1 e somas de quadrados sequenciais
partes <- list(educ = (P1 - P0) %*% y, exper_tenure = (P2 - P1) %*% y,
               residuo = y - P2 %*% y)
data.frame(gl = sapply(list(P1 - P0, P2 - P1, diag(n) - P2), function(M) round(sum(diag(M)))),
           SQ = sapply(partes, sq))
c(soma = sum(sapply(partes, sq)), SQ_total = sq(y - mean(y)))
anova(lm(y ~ educ + exper + tenure, data = w))

# P_[X1 X2] = P_X1 + P_{M1 X2}, com M1 = I - P_X1
Z  <- (diag(n) - P1) %*% cbind(w$exper, w$tenure)
max(abs(P2 - (P1 + proj(Z))))

# Restrição K b = 0 (exper e tenure com o mesmo coeficiente): P_X - P_S = P_{X (X'X)^{-1} K'}
K  <- matrix(c(0, 0, 1, -1), 1)
XR <- cbind(1, w$educ, w$exper + w$tenure)            # C(XR) = {X b : K b = 0}
PR <- proj(XR)
U  <- X2 %*% solve(crossprod(X2), t(K))
c(erro = max(abs(P2 - PR - proj(U))), SQ_restricao = sq((P2 - PR) %*% y))
anova(lm(y ~ educ + I(exper + tenure), data = w), lm(y ~ educ + exper + tenure, data = w))
