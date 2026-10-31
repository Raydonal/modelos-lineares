# Cap. 13 -- Parcelas subdivididas: MQ = MQG, mas os erros-padrão não
a <- read.csv(file.path("dados", "yates_oats.csv"))
a$gen <- factor(a$gen); a$dose <- factor(a$nitro)
a$parcela <- interaction(a$block, a$gen)
a <- a[order(a$parcela, a$nitro), ]           # 18 parcelas x 4 doses
X <- model.matrix(~ block + gen * dose, a); y <- a$yield
n <- nrow(X); A <- solve(crossprod(X))
# Componentes de variância pelos quadrados médios dos dois estratos
qm <- function(f) { r <- resid(lm(f, a)); sum(r^2) }
qm_sub <- qm(yield ~ parcela + dose + gen:dose) / 45  # E = s2
qm_par <- (qm(yield ~ block + gen + dose + gen:dose) -
           qm(yield ~ parcela + dose + gen:dose)) / 10
# QM das parcelas: E = s2 + 4 sp2
s2 <- qm_sub; sp2 <- (qm_par - qm_sub) / 4
round(c(QM_parcelas = qm_par, QM_subparcelas = qm_sub,
        sigma2 = s2, sigma2_p = sp2), 2)
V <- kronecker(diag(18), s2 * diag(4) + sp2)  # Cov(y) estimada
bmq <- drop(A %*% crossprod(X, y))
bg <- drop(solve(t(X) %*% solve(V, X), t(X) %*% solve(V, y)))
c(max_dif_mq_mqg = max(abs(bmq - bg)),        # Col(VX) em Col(X)?
  inclusao = max(abs(V %*% X - X %*% A %*% t(X) %*% V %*% X)))
# Erros-padrão: ingênuo (s2 único) e correto (sanduíche com V)
s2u <- sum((y - X %*% bmq)^2) / (n - ncol(X))
med <- function(g, d) { m <- colMeans(X[a$gen == g &
  (if (is.null(d)) TRUE else a$dose == d), ]); m }
L <- rbind(
  variedade = med("Marvellous", NULL) - med("GoldenRain", NULL),
  dose = (med("GoldenRain", "0.6") + med("Marvellous", "0.6") +
          med("Victory", "0.6") - med("GoldenRain", "0") -
          med("Marvellous", "0") - med("Victory", "0")) / 3)
C <- A %*% t(X) %*% V %*% X %*% A
round(cbind(estimativa = drop(L %*% bmq),
            ep_ingenuo = sqrt(s2u * diag(L %*% A %*% t(L))),
            ep_correto = sqrt(diag(L %*% C %*% t(L)))), 3)
