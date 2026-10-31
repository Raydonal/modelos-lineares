# Cap. 12: Funções estimáveis num fator com três níveis
y <- c(5, 7, 10, 12, 9); grupo <- c(1, 1, 2, 2, 3)
X <- cbind(1, outer(grupo, 1:3, "==") * 1)    # [1 D]: posto 3
XtX <- crossprod(X); Xty <- crossprod(X, y)
G1 <- MASS::ginv(XtX)                         # Moore-Penrose
G2 <- matrix(0, 4, 4)                         # zera o intercepto
G2[2:4, 2:4] <- solve(XtX[2:4, 2:4])
b1 <- drop(G1 %*% Xty); b2 <- drop(G2 %*% Xty)
round(rbind(b_mp = b1, b_ref = b2), 3)  # mu, a1, a2, a3
L <- rbind("mu+a1" = c(1, 1, 0, 0), "a1-a2" = c(0, 1, -1, 0),
           "a1-(a2+a3)/2" = c(0, 1, -.5, -.5),
           "mu" = c(1, 0, 0, 0), "a1" = c(0, 1, 0, 0))
# teste: l'(I - G X'X) = 0 <=> l no espaço linha de X
teste <- apply(L, 1, function(l)
  max(abs(l %*% (diag(4) - G1 %*% XtX))))
round(cbind(estimavel = teste < 1e-10, mp = L %*% b1,
            ref = L %*% b2, var_mp = diag(L %*% G1 %*% t(L)),
            var_ref = diag(L %*% G2 %*% t(L))), 3)
