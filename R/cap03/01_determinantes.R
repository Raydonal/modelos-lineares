# Cap. 3: Determinante pela fórmula de Leibniz e suas propriedades
permutacoes <- function(n) {                   # todas as permutações de 1..n
  if (n == 1) return(matrix(1, 1, 1))
  P <- permutacoes(n - 1)
  do.call(rbind, lapply(1:n, function(k) cbind(k, ifelse(P >= k, P + 1, P))))
}
sinal <- function(s) {                          # (-1)^(número de inversões)
  inv <- sum(outer(seq_along(s), seq_along(s), "<") & outer(s, s, ">"))
  (-1)^inv
}
det_leibniz <- function(A) {
  P <- permutacoes(nrow(A))
  sum(apply(P, 1, function(s) sinal(s) * prod(A[cbind(seq_along(s), s)])))
}

set.seed(3)
A <- matrix(round(rnorm(16), 2), 4); B <- matrix(round(rnorm(16), 2), 4)
c(leibniz = det_leibniz(A), lapack = det(A))
c(det_AB = det(A %*% B), produto = det(A) * det(B))      # det(AB) = det(A) det(B)
c(det_At = det(t(A)), det_A = det(A))                     # det(A') = det(A)
A2 <- A; A2[c(1, 3), ] <- A2[c(3, 1), ]; c(troca = det(A2)) # troca de linhas muda o sinal

# |det| = volume: paralelogramo gerado por (3, 1) e (1, 2)
V <- cbind(c(3, 1), c(1, 2)); c(area = abs(det(V)))
# Gram: sqrt(det(X'X)) = área do paralelogramo gerado por duas colunas em R^3
X <- cbind(c(1, 0, 1), c(0, 2, 1)); sqrt(det(crossprod(X)))
