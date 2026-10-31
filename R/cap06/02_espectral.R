# Cap. 6: Teorema espectral, projeções espectrais, quociente de Rayleigh e Ky Fan
A <- matrix(c(4, 1, 1, 0,
              1, 3, 0, 1,
              1, 0, 3, 1,
              0, 1, 1, 2), 4, byrow = TRUE)
e <- eigen(A, symmetric = TRUE); Q <- e$vectors; lam <- e$values
round(lam, 6)
c(QtQ_menos_I = max(abs(crossprod(Q) - diag(4))), reconstrucao = max(abs(Q %*% diag(lam) %*% t(Q) - A)))

# Projeções espectrais: agrupando autovalores iguais (a menos de arredondamento)
grupos <- split(seq_along(lam), round(lam, 8))
Ps <- lapply(grupos, function(i) Q[, i, drop = FALSE] %*% t(Q[, i, drop = FALSE]))
sapply(Ps, function(P) sum(diag(P)))                      # multiplicidades
mu <- sapply(grupos, function(i) mean(lam[i]))           # autovalores distintos
max(abs(Reduce(`+`, Map(`*`, mu, Ps)) - A))              # A = soma de mu P_mu

# Funções de A: raiz quadrada e inversa
raiz <- Q %*% diag(sqrt(lam)) %*% t(Q)
c(raiz2_menos_A = max(abs(raiz %*% raiz - A)), inversa = max(abs(Q %*% diag(1 / lam) %*% t(Q) - solve(A))))

# Quociente de Rayleigh em 10 000 vetores ao acaso fica em [lambda_min, lambda_max]
source(file.path("R", "comum", "gerador.R")); g <- gerador(62)
X <- matriz_normal(g, 10000, 4)
r <- rowSums((X %*% A) * X) / rowSums(X^2)
c(min = min(r), max = max(r), lambda_min = min(lam), lambda_max = max(lam))

# Ky Fan: tr(V'AV) <= lambda_1 + lambda_2 para V com 2 colunas ortonormais
tr2 <- replicate(2000, { V <- qr.Q(qr(matriz_normal(g, 4, 2))); sum(diag(t(V) %*% A %*% V)) })
c(max_ao_acaso = max(tr2), lambda1_mais_lambda2 = sum(lam[1:2]))
