# Cap. 8: Produto de Kronecker, vec e vech
source(file.path("R", "comum", "gerador.R")); g <- gerador(81)
A <- matriz_normal(g, 2, 3); B <- matriz_normal(g, 4, 2); C <- matriz_normal(g, 3, 5); D <- matriz_normal(g, 2, 3)
c(produto_misto = max(abs(kronecker(A, B) %*% kronecker(C, D) - kronecker(A %*% C, B %*% D))),
  transposta = max(abs(t(kronecker(A, B)) - kronecker(t(A), t(B)))))
P <- crossprod(matriz_normal(g, 3, 3)) + diag(3); Q <- crossprod(matriz_normal(g, 2, 2)) + diag(2)
K <- kronecker(P, Q)
lp <- eigen(P)$values; lq <- eigen(Q)$values
c(autovalores = max(abs(sort(eigen(K)$values) - sort(outer(lp, lq)))),
  det = det(K) / (det(P)^2 * det(Q)^3), traco = sum(diag(K)) - sum(diag(P)) * sum(diag(Q)),
  inversa = max(abs(solve(K) - kronecker(solve(P), solve(Q)))))
# vec(A X C) = (C' kron A) vec(X), e a solução de A X C = B (Cap. 4)
vec <- function(M) as.vector(M)
X <- matriz_normal(g, 3, 3); A1 <- matriz_normal(g, 4, 3); C1 <- matriz_normal(g, 3, 2)
max(abs(vec(A1 %*% X %*% C1) - kronecker(t(C1), A1) %*% vec(X)))
Bm <- A1 %*% X %*% C1
Xs <- MASS::ginv(A1) %*% Bm %*% MASS::ginv(C1)              # solução de norma mínima
c(resolve = max(abs(A1 %*% Xs %*% C1 - Bm)),
  mesma_que_via_vec = max(abs(vec(Xs) - MASS::ginv(kronecker(t(C1), A1)) %*% vec(Bm))))
# Matriz de comutação K: vec(A') = K vec(A); matriz de duplicação D: vec(S) = D vech(S)
comutacao <- function(m, n) { Km <- matrix(0, m * n, m * n)
  for (i in 1:m) for (j in 1:n) Km[(i - 1) * n + j, (j - 1) * m + i] <- 1; Km }
Am <- matriz_normal(g, 2, 3)
max(abs(comutacao(2, 3) %*% vec(Am) - vec(t(Am))))
duplicacao <- function(n) { Dm <- matrix(0, n * n, n * (n + 1) / 2); k <- 0
  for (j in 1:n) for (i in j:n) { k <- k + 1; Dm[(j - 1) * n + i, k] <- 1; Dm[(i - 1) * n + j, k] <- 1 }; Dm }
vech <- function(M) M[lower.tri(M, diag = TRUE)]
c(duplicacao = max(abs(duplicacao(3) %*% vech(P) - vec(P))), dim = dim(duplicacao(3)))
