# Cap. 10 -- Deriva de embeddings detectada pelo T^2 de Hotelling
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
source(file.path("R", "comum", "austen.R"))
emb <- embeddings_austen(); E <- emb$E
d <- read.csv(file.path("dados", "austen.csv"))
blocos <- function(livro, L = 200) {          # médias de 200 palavras
  txt <- tolower(d$texto[d$livro == livro])
  ids <- match(unlist(regmatches(txt, gregexpr("[a-z]+", txt))),
               emb$vocab)
  ids <- ids[!is.na(ids)]; m <- length(ids) %/% L
  rowsum(E[ids[1:(m * L)], ], rep(1:m, each = L)) / L
}
pp <- blocos("Pride & Prejudice"); h <- nrow(pp) %/% 2
ref <- pp[1:h, ]
mu <- colMeans(ref); p <- 10
V <- svd(sweep(ref, 2, mu))$v[, 1:p]          # PCA da referência
t2 <- function(X1, X2) {
  Z1 <- sweep(X1, 2, mu) %*% V; Z2 <- sweep(X2, 2, mu) %*% V
  n1 <- nrow(Z1); n2 <- nrow(Z2); g <- n1 + n2 - 2
  dm <- colMeans(Z1) - colMeans(Z2)
  Sp <- ((n1 - 1) * cov(Z1) + (n2 - 1) * cov(Z2)) / g
  T2 <- n1 * n2 / (n1 + n2) * drop(t(dm) %*% solve(Sp, dm))
  Fst <- T2 * (g - p + 1) / (p * g)
  c(T2 = T2, F = Fst, p = pf(Fst, p, g - p + 1, lower.tail = FALSE))
}
lote <- list(PP_2a_metade = pp[(h + 1):(h + 30), ],
             Emma = blocos("Emma")[1:30, ],
             Persuasion = blocos("Persuasion")[1:30, ],
             Sense = blocos("Sense & Sensibility")[1:30, ])
signif(t(sapply(lote, function(X) t2(ref, X))), 4)
# Calibração: 500 partições ao acaso da referência
g <- gerador(41)
rej <- replicate(500, {
  o <- order(g$unif(h)); k <- h %/% 2
  t2(ref[o[1:k], ], ref[o[-(1:k)], ])["p"] < 0.05
})
c(n_referencia = h, falsos_positivos = mean(rej))
