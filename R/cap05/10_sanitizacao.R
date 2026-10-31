# Cap. 5 -- Sanitização de embeddings: projetar fora o gênero
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
source(file.path("R", "comum", "austen.R"))   # embeddings do Cap. 6
emb <- embeddings_austen(); E <- emb$E; C <- emb$C; vocab <- emb$vocab
pos <- function(w) match(w, vocab)

# 1. Direção de gênero: média das diferenças em pares definidores
pares <- matrix(c("he", "she", "his", "her", "himself", "herself",
  "mr", "mrs", "sir", "lady", "man", "woman", "men", "women",
  "gentleman", "lady", "gentlemen", "ladies", "brother", "sister",
  "brothers", "sisters", "father", "mother", "son", "daughter",
  "husband", "wife", "uncle", "aunt", "nephew", "niece",
  "boy", "girl"), ncol = 2, byrow = TRUE)
v <- colMeans(E[pos(pares[, 2]), ] - E[pos(pares[, 1]), ])
v <- v / sqrt(sum(v^2))
nomes <- c("darcy", "edmund", "knightley", "thomas", "john",
  "henry", "edward", "william", "frank", "wickham", "charles",
  "willoughby", "brandon", "walter", "james", "tom", "robert",
  "frederick", "bingley", "collins", "wentworth", "tilney",
  "rushworth", "benwick", "fanny", "emma", "elizabeth", "elinor",
  "catherine", "jane", "marianne", "anne", "harriet", "mary",
  "isabella", "lucy", "lydia", "maria", "julia", "louisa",
  "charlotte", "lizzy", "eleanor", "kitty", "susan", "henrietta",
  "margaret", "eliza")
y_n <- rep(c(-1, 1), each = 24)        # -1 masculino, +1 feminino
c(acerto_nomes_por_v = mean(sign(E[pos(nomes), ] %*% v) == y_n))

# 2. Rótulos: palavras que coocorrem mais com she/her ou he/his
cf <- C[, pos("she")] + C[, pos("her")]
cm <- C[, pos("he")] + C[, pos("his")]
r <- log((cf + 1) / (cm + 1)) - log(sum(cf) / sum(cm))
o <- setdiff(order(r), pos(c(pares, nomes)))
K <- 300; sel <- c(head(o, K), rev(o)[1:K])
y <- rep(c(-1, 1), each = K)
cat("mais masculinas:", vocab[head(o, 8)], "\n")
cat("mais femininas: ", vocab[rev(o)[1:8]], "\n")
g <- gerador(7); perm <- order(g$unif(2 * K))
tr <- sel[perm[1:420]]; y_tr <- y[perm[1:420]]
te <- sel[perm[421:600]]; y_te <- y[perm[421:600]]

# 3. INLP: classificador de MQ em C(Q)^perp; remove sua direção
complemento <- function(Q) {
  if (ncol(Q) == 0) return(diag(ncol(E)))
  qr.Q(qr(Q), complete = TRUE)[, -(1:ncol(Q)), drop = FALSE]
}
knn <- function(Z, k = 5) {            # k vizinhos entre os nomes
  Z <- Z / sqrt(rowSums(Z^2)); S <- Z %*% t(Z); diag(S) <- -Inf
  viz <- t(apply(S, 1, order, decreasing = TRUE))[, 1:k]
  mean(sign(rowSums(matrix(y_n[viz], ncol = k))) == y_n)
}
avaliar <- function(Q) {
  B <- complemento(Q); Z <- E[tr, ] %*% B; mu <- colMeans(Z)
  gam <- qr.coef(qr(sweep(Z, 2, mu)), y_tr)
  prev <- function(i) sign(sweep(E[i, ] %*% B, 2, mu) %*% gam)
  list(w = B %*% gam / sqrt(sum(gam^2)),
       acc = c(palavras = mean(prev(te) == y_te),
               nomes = mean(prev(pos(nomes)) == y_n),
               nomes_knn = knn(E[pos(nomes), ] %*% B)))
}
Q <- matrix(0, ncol(E), 0); tab <- NULL
for (r_ in 0:40) {
  a <- avaliar(Q)
  if (r_ %in% c(0, 1, 2, 3, 5, 10, 20, 40))
    tab <- rbind(tab, c(r = r_, a$acc))
  if (r_ == 10) Q10 <- Q               # as 10 primeiras direções
  Q <- cbind(Q, if (r_ == 0) v else a$w)  # 1a: v; depois, INLP
}
round(tab, 3)
max(abs(crossprod(Q) - diag(ncol(Q))))  # direções ortonormais

# 4. O que a projeção P = I - QQ' (r = 10) preserva
P <- diag(ncol(E)) - Q10 %*% t(Q10)
c(norma_removida = mean(rowSums((E %*% Q10)^2)),
  cor_cossenos = cor(as.vector(tcrossprod(E)),
                     as.vector(E %*% P %*% t(E))))
Es <- E %*% P; Es <- Es / sqrt(rowSums(Es^2))
viz <- function(M, w) vocab[order(-drop(M %*% M[pos(w), ]))[2:6]]
for (w in c("happy", "letter", "london", "mother", "elizabeth"))
  cat(sprintf("%-9s antes: %s\n%-9s depois: %s\n", w,
              paste(viz(E, w), collapse = ", "), "",
              paste(viz(Es, w), collapse = ", ")))
