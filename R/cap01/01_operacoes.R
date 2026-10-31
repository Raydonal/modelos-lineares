# Cap. 1 -- Operações básicas com vetores e matrizes
A <- matrix(c(2, 1,
              0, 3,
              1, -1), nrow = 3, byrow = TRUE)    # 3 x 2
B <- matrix(c(1, 4, 2,
              0, 1, 5), nrow = 2, byrow = TRUE)  # 2 x 3
x <- c(2, -1)

A %*% B                          # produto 3 x 3
t(A %*% B) - t(B) %*% t(A)       # (AB)' = B'A'  -> matriz nula
sum(diag(A %*% B)) - sum(diag(B %*% A))   # tr(AB) = tr(BA) -> 0

# Ax como combinação linear das colunas de A
A %*% x
x[1] * A[, 1] + x[2] * A[, 2]

# Produto como soma de produtos externos: AB = sum_k a_k b_k'
Reduce(`+`, lapply(1:2, function(k) A[, k] %o% B[k, ]))
