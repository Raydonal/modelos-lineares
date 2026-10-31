# Cap. 10 -- Cochran na análise de variância com um fator
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
a <- 3; r <- 5; n <- a * r; sigma <- 2
grupo <- rep(1:a, each = r)
D <- outer(grupo, 1:a, "==") * 1
J <- matrix(1 / n, n, n); P <- D %*% solve(crossprod(D)) %*% t(D)
A1 <- P - J; A2 <- diag(n) - P               # entre e dentro
c(postos = c(sum(diag(A1)), sum(diag(A2))),
  A1A2 = max(abs(A1 %*% A2)))
mu <- rep(c(10, 11, 13), each = r)          # médias dos grupos
lambda <- drop(t(mu) %*% A1 %*% mu) / sigma^2
g <- gerador(31); R <- 20000
Y <- sweep(matriz_normal(g, R, n, sigma), 2, mu, "+")
q1 <- rowSums((Y %*% A1) * Y) / sigma^2     # SQ entre / sigma^2
q2 <- rowSums((Y %*% A2) * Y) / sigma^2     # SQ dentro / sigma^2
F <- (q1 / (a - 1)) / (q2 / (n - a))
c(lambda = lambda, media_q1 = mean(q1), teorica = a - 1 + lambda,
  media_q2 = mean(q2), teorica = n - a, correlacao = cor(q1, q2),
  poder_sim = mean(F > qf(0.95, a - 1, n - a)),
  poder_teo = pf(qf(0.95, a - 1, n - a), a - 1, n - a, ncp = lambda,
                 lower.tail = FALSE))
