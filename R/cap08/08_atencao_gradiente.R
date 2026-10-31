# Cap. 8: Retropropagação numa camada de atenção, conferida por diferenças finitas
source(file.path("R", "comum", "gerador.R")); g <- gerador(88)
n <- 6; d <- 5; dk <- 3
X <- matriz_normal(g, n, d); Tg <- matriz_normal(g, n, d)  # entrada e alvo
Wq <- matriz_normal(g, d, dk); Wk <- matriz_normal(g, d, dk); Wv <- matriz_normal(g, d, d)
softmax <- function(S) { S <- S - apply(S, 1, max); E <- exp(S); E / rowSums(E) }
frente <- function(Wq, Wk, Wv) {
  Q <- X %*% Wq; K <- X %*% Wk; V <- X %*% Wv
  A <- softmax(Q %*% t(K) / sqrt(dk)); O <- A %*% V
  list(Q = Q, K = K, V = V, A = A, O = O, perda = 0.5 * sum((O - Tg)^2))
}
f <- frente(Wq, Wk, Wv)
# Retropropagação (regras da Seção de cálculo matricial)
GO <- f$O - Tg                                             # dL/dO
GV <- t(f$A) %*% GO; GWv <- t(X) %*% GV                    # O = A V
GA <- GO %*% t(f$V)
GS <- f$A * (GA - rowSums(GA * f$A))                       # softmax por linha: a .* (g - (a'g) 1)
GQ <- GS %*% f$K / sqrt(dk); GK <- t(GS) %*% f$Q / sqrt(dk)
GWq <- t(X) %*% GQ; GWk <- t(X) %*% GK
# Diferenças finitas, entrada por entrada
num <- function(W, qual) { Gm <- W * 0; h <- 1e-6
  for (i in seq_along(W)) { Wp <- W; Wm <- W; Wp[i] <- Wp[i] + h; Wm[i] <- Wm[i] - h
    args_p <- list(Wq, Wk, Wv); args_m <- list(Wq, Wk, Wv); args_p[[qual]] <- Wp; args_m[[qual]] <- Wm
    Gm[i] <- (do.call(frente, args_p)$perda - do.call(frente, args_m)$perda) / (2 * h) }; Gm }
c(Wq = max(abs(num(Wq, 1) - GWq)), Wk = max(abs(num(Wk, 2) - GWk)), Wv = max(abs(num(Wv, 3) - GWv)))
# Jacobiano da softmax: diag(a) - a a' é simétrico, semidefinido, anula 1 e tem norma <= 1/2
Js <- lapply(1:n, function(i) { a <- f$A[i, ]; diag(a) - a %o% a })
c(max_norma = max(sapply(Js, function(J) max(eigen(J, symmetric = TRUE)$values))),
  min_autovalor = min(sapply(Js, function(J) min(eigen(J, symmetric = TRUE)$values))),
  J_vezes_1 = max(sapply(Js, function(J) max(abs(J %*% rep(1, n))))))
# Logits extremos: a norma 1/2 é atingida quando dois tokens dividem a atenção
a <- c(0.5, 0.5, 0, 0); max(eigen(diag(a) - a %o% a, symmetric = TRUE)$values)
