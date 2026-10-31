# Cap. 7 -- Perturbação de autovalores e valores singulares; Eckart-Young na norma espectral
source(file.path("R", "comum", "gerador.R")); g <- gerador(74)
S <- matriz_normal(g, 6, 6); A <- S + t(S)                # simétrica
E <- matriz_normal(g, 6, 6); E <- 1e-3 * (E + t(E))       # perturbação simétrica
la <- eigen(A, symmetric = TRUE)$values; lb <- eigen(A + E, symmetric = TRUE)$values
c(max_variacao_autovalor = max(abs(lb - la)), norma2_E = svd(E)$d[1])
X <- matriz_normal(g, 8, 5); F <- 1e-3 * matriz_normal(g, 8, 5)
c(max_variacao_sigma = max(abs(svd(X + F)$d - svd(X)$d)), norma2_F = svd(F)$d[1])
# Eckart-Young na norma espectral e distância à matriz singular mais próxima
s <- svd(X)
erro2 <- sapply(1:4, function(k) svd(X - s$u[, 1:k] %*% diag(s$d[1:k], k) %*% t(s$v[, 1:k]))$d[1])
rbind(k = 1:4, erro_espectral = round(erro2, 6), sigma_k_mais_1 = round(s$d[2:5], 6))
M <- matriz_normal(g, 4, 4); sm <- svd(M)
Msing <- M - sm$d[4] * sm$u[, 4] %*% t(sm$v[, 4])        # a matriz singular mais próxima
c(sigma_min = sm$d[4], distancia = svd(M - Msing)$d[1], det_Msing = det(Msing))
