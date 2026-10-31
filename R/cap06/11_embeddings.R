# Cap. 6 -- Embeddings de palavras pela SVD da matriz PPMI de coocorrências (romances de Jane Austen)
d <- read.csv(file.path("dados", "austen.csv"))
p <- unlist(regmatches(tolower(d$texto), gregexpr("[a-z]+", tolower(d$texto))))
length(p)                                              # palavras no corpus
fr <- table(p); fr <- fr[order(-fr, names(fr), method = "radix")]
V <- 2000; vocab <- names(fr)[1:V]                     # as 2000 palavras mais frequentes
ids <- match(p, vocab)
# Coocorrências numa janela de 2 palavras de cada lado
C <- matrix(0, V, V)
for (o in 1:2) {
  i <- ids[1:(length(ids) - o)]; j <- ids[(1 + o):length(ids)]
  ok <- !is.na(i) & !is.na(j)
  C <- C + matrix(tabulate(i[ok] + (j[ok] - 1) * V, V * V), V, V)
}
C <- C + t(C)                                          # janela simétrica
# PPMI: max(0, log P(i,j) / (P(i) P_alfa(j))), com a suavização alfa = 0,75 do contexto
N <- sum(C); pi_ <- rowSums(C) / N
pc <- colSums(C)^0.75; pc <- pc / sum(pc)
M <- log((C / N) / outer(pi_, pc)); M[!is.finite(M) | M < 0] <- 0
c(densidade = mean(M > 0))
s <- svd(M)
round(s$d[c(1, 2, 10, 50, 100, 500)], 2)
k <- 100
c(fracao_frobenius_k = sum(s$d[1:k]^2) / sum(s$d^2))
W <- s$u[, 1:k] %*% diag(sqrt(s$d[1:k]))               # embeddings de dimensão 100
rownames(W) <- vocab
Wn <- W / sqrt(rowSums(W^2))
vizinhos <- function(w, m = 6) {
  cs <- drop(Wn %*% Wn[w, ]); names(sort(cs, decreasing = TRUE)[2:(m + 1)])
}
for (w in c("elizabeth", "happy", "letter", "mother", "walked", "london", "ten"))
  cat(sprintf("%-10s %s\n", w, paste(vizinhos(w), collapse = ", ")))
# Mesma busca na matriz PPMI completa (2000 dimensões, esparsa e ruidosa)
Mn <- M / sqrt(rowSums(M^2)); rownames(Mn) <- vocab
viz_M <- function(w, m = 6) names(sort(drop(Mn %*% Mn[w, ]), decreasing = TRUE)[2:(m + 1)])
for (w in c("happy", "walked")) cat(sprintf("%-10s %s\n", w, paste(viz_M(w), collapse = ", ")))
