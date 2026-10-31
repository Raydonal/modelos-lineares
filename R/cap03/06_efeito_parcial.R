# Cap. 3: Coeficiente parcial pela inversa particionada (prévia de Frisch-Waugh-Lovell)
dados <- read.csv(file.path("dados", "wage1.csv"))
y <- log(dados$wage)
X1 <- cbind(1, exper = dados$exper)              # controles
x2 <- dados$educ                                  # variável de interesse
X <- cbind(X1, educ = x2)

b <- solve(crossprod(X), crossprod(X, y))          # regressão completa
# Bloco (2,2) da inversa particionada = 1 / complemento de Schur s = x2'x2 - x2'X1 (X1'X1)^{-1} X1'x2
s <- sum(x2^2) - crossprod(x2, X1) %*% solve(crossprod(X1), crossprod(X1, x2))
c(bloco_inversa = solve(crossprod(X))[3, 3], inverso_schur = 1 / s)

# s é a soma de quadrados do resíduo de educ regredida nos controles
e2 <- x2 - X1 %*% solve(crossprod(X1), crossprod(X1, x2))
c(schur = s, sq_residuo = sum(e2^2))
c(b_educ_completo = b[3], b_educ_residuo = sum(e2 * y) / sum(e2^2),
  b_educ_simples = cov(x2, y) / var(x2))
