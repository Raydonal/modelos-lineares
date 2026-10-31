# Cap. 7 -- Quantização de embeddings: erro em normas e preservação de vizinhos
d <- read.csv(file.path("dados", "austen.csv"))
p <- unlist(regmatches(tolower(d$texto), gregexpr("[a-z]+", tolower(d$texto))))
fr <- table(p); fr <- fr[order(-fr, names(fr), method = "radix")]
V <- 2000; vocab <- names(fr)[1:V]; ids <- match(p, vocab)
C <- matrix(0, V, V)
for (o in 1:2) { i <- ids[1:(length(ids) - o)]; j <- ids[(1 + o):length(ids)]; ok <- !is.na(i) & !is.na(j)
  C <- C + matrix(tabulate(i[ok] + (j[ok] - 1) * V, V * V), V, V) }
C <- C + t(C); N <- sum(C); pc <- colSums(C)^0.75; pc <- pc / sum(pc)
M <- log((C / N) / outer(rowSums(C) / N, pc)); M[!is.finite(M) | M < 0] <- 0
s <- svd(M, nu = 100, nv = 0)
W <- s$u %*% diag(sqrt(s$d[1:100]))                        # embeddings do Cap. 6 (2000 x 100)
c(max_abs = max(abs(W)), quantil_99 = unname(quantile(abs(W), 0.99)))
quantiza <- function(W, bits, por_linha) {                 # arredondamento simétrico em 2^(bits-1)-1 níveis
  q <- 2^(bits - 1) - 1
  esc <- if (por_linha) apply(abs(W), 1, max) / q else max(abs(W)) / q
  round(W / esc) * esc
}
vizinhos <- function(W, k = 5) {                           # 5 vizinhos (cosseno) das 300 palavras mais frequentes
  nr <- sqrt(rowSums(W^2)); nr[nr == 0] <- 1; Wn <- W / nr
  S <- Wn[1:300, ] %*% t(Wn); S[cbind(1:300, 1:300)] <- -Inf
  t(apply(S, 1, function(v) order(-v)[1:k]))
}
v0 <- vizinhos(W); n2 <- function(A) svd(A)$d[1]
res <- t(sapply(list(c(8, 1), c(8, 0), c(4, 1), c(4, 0), c(3, 1), c(2, 1)), function(cfg) {
  Q <- quantiza(W, cfg[1], cfg[2] == 1); E <- W - Q; vq <- vizinhos(Q)
  c(bits = cfg[1], por_linha = cfg[2], erro_espectral = n2(E) / n2(W), erro_frobenius = norm(E, "F") / norm(W, "F"),
    vizinhos_preservados = mean(sapply(1:300, function(i) length(intersect(v0[i, ], vq[i, ])) / 5)))
}))
round(res, 4)
# Cota: |cos(q_i, q_j) - cos(w_i, w_j)| <= 2 (||e_i||/||w_i|| + ||e_j||/||w_j||); pior erro relativo por linha
Q4 <- quantiza(W, 4, TRUE)
round(c(max_erro_rel_linha = max(sqrt(rowSums((W - Q4)^2)) / sqrt(rowSums(W^2)))), 4)
