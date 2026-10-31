# Cap. 6: Componentes principais e quase colinearidade nos dados de Longley
L <- read.csv(file.path("dados", "longley.csv"))
X <- as.matrix(L[, c("GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year")])
R <- cor(X)                                     # matriz de correlações (variáveis padronizadas)
e <- eigen(R, symmetric = TRUE)
round(e$values, 5)
round(cumsum(e$values) / ncol(X), 4)            # proporção acumulada da variância
c(kappa = max(e$values) / min(e$values))        # número de condição de R
V <- e$vectors; rownames(V) <- colnames(X)
round(V[, c(1, 2, 6)], 3)                       # 1a, 2a e última componentes

# A última componente é quase constante: a combinação v6' z tem variância lambda_6
Z <- scale(X)
c(var_ultima = var(drop(Z %*% V[, 6])), lambda6 = e$values[6])
# Fatores de inflação da variância: diagonal de R^{-1} = sum_j v_ij^2 / lambda_j
vif <- diag(solve(R))
round(rbind(vif = vif, espectral = rowSums(V^2 %*% diag(1 / e$values))), 1)
# Contribuição da última componente para cada VIF
round(V[, 6]^2 / e$values[6] / vif, 3)
