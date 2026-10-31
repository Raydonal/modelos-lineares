# Embeddings de palavras dos romances de Jane Austen pela SVD da matriz PPMI
# (construção explicada no Cap. 6, R/cap06/11_embeddings.R). Devolve o
# vocabulário, as coocorrências C e os embeddings normalizados E (k colunas).
embeddings_austen <- function(V = 2000, k = 100) {
  d <- read.csv(file.path("dados", "austen.csv"))
  txt <- tolower(d$texto)
  p <- unlist(regmatches(txt, gregexpr("[a-z]+", txt)))
  fr <- table(p); fr <- fr[order(-fr, names(fr), method = "radix")]
  vocab <- names(fr)[1:V]; ids <- match(p, vocab)
  C <- matrix(0, V, V)
  for (o in 1:2) {
    i <- ids[1:(length(ids) - o)]; j <- ids[(1 + o):length(ids)]
    ok <- !is.na(i) & !is.na(j)
    C <- C + matrix(tabulate(i[ok] + (j[ok] - 1) * V, V * V), V, V)
  }
  C <- C + t(C)
  N <- sum(C); pi_ <- rowSums(C) / N
  pc <- colSums(C)^0.75; pc <- pc / sum(pc)
  M <- log((C / N) / outer(pi_, pc)); M[!is.finite(M) | M < 0] <- 0
  s <- svd(M, nu = k, nv = 0)
  W <- s$u %*% diag(sqrt(s$d[1:k]))
  list(vocab = vocab, C = C, E = W / sqrt(rowSums(W^2)))
}
