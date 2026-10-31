# Cap. 4: Equações normais de um modelo superparametrizado: o que muda e o que não muda
grupo <- factor(c(1, 1, 2, 2, 2, 3, 3))
y <- c(12, 14, 18, 17, 19, 10, 11)
X <- cbind(1, model.matrix(~ grupo - 1))          # mu, alfa1, alfa2, alfa3 (posto 3)
XtX <- crossprod(X); Xty <- crossprod(X, y)

Gmp  <- MASS::ginv(XtX)                            # Moore-Penrose
Gsub <- matrix(0, 4, 4); Gsub[2:4, 2:4] <- solve(XtX[2:4, 2:4])   # zera a linha/coluna de mu
c(erro_mp = max(abs(XtX %*% Gmp %*% XtX - XtX)), erro_sub = max(abs(XtX %*% Gsub %*% XtX - XtX)))

b_mp <- Gmp %*% Xty; b_sub <- Gsub %*% Xty        # duas soluções das equações normais
round(cbind(mp = drop(b_mp), sub = drop(b_sub)), 4)

H_mp <- X %*% Gmp %*% t(X); H_sub <- X %*% Gsub %*% t(X)
c(dif_H = max(abs(H_mp - H_sub)), simetria = max(abs(H_mp - t(H_mp))),
  idempotencia = max(abs(H_mp %*% H_mp - H_mp)), traco = sum(diag(H_mp)))

# lambda' b: invariante se lambda' está no espaço linha de X
L <- rbind(mu = c(1, 0, 0, 0), alfa1 = c(0, 1, 0, 0),
           mu_alfa1 = c(1, 1, 0, 0), alfa1_alfa2 = c(0, 1, -1, 0))
cbind(mp = drop(L %*% b_mp), sub = drop(L %*% b_sub))
