# Cap. 8 -- Derivadas matriciais conferidas por diferenças finitas
source(file.path("R", "comum", "gerador.R")); g <- gerador(80)
n <- 4
X <- crossprod(matriz_normal(g, 6, n)) + diag(n)            # definida positiva
dX <- matriz_normal(g, n, n); h <- 1e-6
df <- function(F) (F(X + h * dX) - F(X - h * dX)) / (2 * h)  # derivada direcional central
tr <- function(M) sum(diag(M))
Xi <- solve(X)
c(log_det = df(function(M) log(det(M))) - tr(Xi %*% dX),
  det_relativo = (df(det) - det(X) * tr(Xi %*% dX)) / det(X),
  inversa = max(abs(df(solve) + Xi %*% dX %*% Xi)),
  traco_AX = df(function(M) tr(diag(1:n) %*% M)) - tr(diag(1:n) %*% dX))
# Gradientes: d/dX log det X = X^{-T};  d/dX det X = matriz de cofatores
grad_num <- function(F) { Gm <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) { E <- matrix(0, n, n); E[i, j] <- h; Gm[i, j] <- (F(X + E) - F(X - E)) / (2 * h) }; Gm }
c(grad_logdet = max(abs(grad_num(function(M) log(det(M))) - t(Xi))),
  grad_det_cofatores_rel = max(abs(grad_num(det) - det(X) * t(Xi))) / det(X))
# Formas quadráticas: d(x'Ax)/dx = (A + A')x
A <- matriz_normal(g, n, n); x <- drop(matriz_normal(g, n, 1))
gq <- sapply(1:n, function(i) { e <- replace(numeric(n), i, h); (drop(t(x + e) %*% A %*% (x + e)) - drop(t(x - e) %*% A %*% (x - e))) / (2 * h) })
max(abs(gq - (A + t(A)) %*% x))
# Derivada de um autovalor simples: d lambda = q' dA q
S <- X; dS <- dX + t(dX)
lam <- function(M) eigen(M, symmetric = TRUE)$values
q1 <- eigen(S, symmetric = TRUE)$vectors[, 1]
c(numerica = (lam(S + h * dS)[1] - lam(S - h * dS)[1]) / (2 * h), formula = drop(t(q1) %*% dS %*% q1))
