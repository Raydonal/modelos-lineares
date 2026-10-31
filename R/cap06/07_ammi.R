# Cap. 6: Interação genótipo x ambiente pela SVD: o modelo AMMI (trigo de inverno, Ontário, 1993)
w <- read.csv(file.path("dados", "yan_trigo.csv"))
Y <- with(w, tapply(producao, list(genotipo, ambiente), mean))   # 18 genótipos x 9 ambientes
dim(Y)
m <- mean(Y); a <- rowMeans(Y) - m; b <- colMeans(Y) - m
I <- Y - m - outer(a, rep(1, ncol(Y))) - outer(rep(1, nrow(Y)), b)   # interação (duplo centrado)
c(soma_linhas = max(abs(rowSums(I))), soma_colunas = max(abs(colSums(I))))
SQ <- c(genotipos = ncol(Y) * sum(a^2), ambientes = nrow(Y) * sum(b^2), interacao = sum(I^2))
round(rbind(SQ = SQ, proporcao = SQ / sum((Y - m)^2)), 3)

s <- svd(I)
round(s$d, 3)
c(posto = sum(s$d > 1e-10), maximo = min(dim(Y)) - 1)
round(rbind(proporcao = s$d^2 / sum(s$d^2), acumulada = cumsum(s$d^2) / sum(s$d^2)), 3)
# Escores AMMI2 (sigma_k^{1/2} u_k e sigma_k^{1/2} v_k)
G <- s$u[, 1:2] %*% diag(sqrt(s$d[1:2])); E <- s$v[, 1:2] %*% diag(sqrt(s$d[1:2]))
dimnames(G) <- list(rownames(Y), c("CP1", "CP2")); dimnames(E) <- list(colnames(Y), c("CP1", "CP2"))
round(E, 2)
round(G[order(-abs(G[, 1]))[1:5], ], 2)
# Eckart-Young: o erro do AMMI2 na interação é a soma dos sigma^2 descartados
I2 <- G %*% t(E)
c(erro = sum((I - I2)^2), descartados = sum(s$d[-(1:2)]^2))
# Previsão AMMI2 para o genótipo de maior escore no ambiente de menor escore
c(observado = Y["Luc", "OA93"], aditivo = m + a["Luc"] + b["OA93"], ammi2 = m + a["Luc"] + b["OA93"] + I2["Luc", "OA93"])
