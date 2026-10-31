# Cap. 14: Dígitos: muitos classificadores, um só teste
source(file.path("R", "comum", "gerador.R"))
source(file.path("R", "comum", "multiplas.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
dig <- read.csv(file.path("dados", "digitos.csv"))
P <- as.matrix(dig[, -1]) / 16; dimnames(P) <- NULL; rot <- dig$digito
tr <- 1:1200; te <- 1201:nrow(P)                 # 597 imagens
med <- colMeans(P[tr, ]); Xc <- sweep(P, 2, med)
Xt <- Xc[tr, ]; Y <- outer(rot[tr], 0:9, "==") * 1
ev <- eigen(crossprod(Xt), symmetric = TRUE)$vectors
pca <- function(k) Xc %*% ev[, 1:k, drop = FALSE]
centr <- function(Sc) {                         # centróide euclidiano
  C <- rowsum(Sc[tr, ], rot[tr]) / as.vector(table(rot[tr]))
  d <- sapply(1:10, function(j)
    rowSums(sweep(Sc[te, , drop = FALSE], 2, C[j, ])^2))
  max.col(-d, ties.method = "first") - 1
}
lin <- function(Sc, lam = 0) {                  # regressão linear
  St <- Sc[tr, ]; Yc <- sweep(Y, 2, colMeans(Y))
  B <- solve(crossprod(St) + lam * diag(ncol(St)), crossprod(St, Yc))
  max.col(Sc[te, ] %*% B, ties.method = "first") - 1
}
knn <- function(k) {                            # k vizinhos
  sapply(te, function(i) {
    d <- rowSums(sweep(P[tr, ], 2, P[i, ])^2)
    v <- rot[tr][order(d)[1:k]]
    which.max(tabulate(v + 1, 10)) - 1
  })
}
C <- list(cent = centr(Xc))
for (k in c(10, 20, 40)) C[[paste0("cent", k)]] <- centr(pca(k))
for (k in c(10, 20, 30, 40, 50))
  C[[paste0("lin", k)]] <- lin(pca(k))
for (l in c(0.01, 0.1, 1, 10, 100, 1000))
  C[[paste0("ridge", l)]] <- lin(Xc, l)
for (k in c(1, 3, 5, 7, 9)) C[[paste0("knn", k)]] <- knn(k)
Ac <- sapply(C, function(p) (p == rot[te]) * 1)
# Validação e teste final sorteados entre as 597 imagens
g <- gerador(597); o <- order(g$unif(597))
val <- o[1:300]; fim <- o[301:597]; m <- ncol(Ac) - 1
acv <- colMeans(Ac[val, ]); acf <- colMeans(Ac[fim, ])
b0 <- match("lin30", names(C))
pr("candidatos %d; base lin30: val %.4f fim %.4f", m, acv[b0],
   acf[b0])
for (j in order(-acv)) pr("%-9s val %.4f fim %.4f",
  names(C)[j], acv[j], acf[j])
# McNemar: d = acerto do candidato - acerto da base
D <- Ac[val, -b0] - Ac[val, b0]
nd <- sqrt(colSums(D^2)); nd[nd == 0] <- 1
tt <- colSums(D) / nd
pv <- 2 * pnorm(-abs(tt))
g <- gerador(1998); N <- 20000
E <- matrix(2 * (g$unif(N * 300) < 0.5) - 1, N, 300, byrow = TRUE)
Tst <- abs(E %*% sweep(D, 2, nd, "/"))
cmax <- sort(apply(Tst, 1, max))[0.95 * N]
pr("t min %.2f max %.2f; crítico max-t %.3f (Bonf %.3f)", min(tt),
   max(tt), cmax, qnorm(1 - 0.025 / m))
pr("achados: sem ajuste %d Bonf %d max-t %d BH %d",
   sum(abs(tt) > qnorm(0.975)), sum(pv <= 0.05 / m),
   sum(abs(tt) > cmax), bh(pv, 0.05))
pr("sem ajuste: %s", paste(colnames(D)[abs(tt) > qnorm(0.975)],
                          collapse = " "))
pr("melhores que a base: Bonf %d max-t %d BH %d",
   sum(pv <= 0.05 / m & tt > 0), sum(abs(tt) > cmax & tt > 0),
   sum(pv <= 0.05 * bh(pv, 0.05) / m & tt > 0))
Rm <- cor(D[, colSums(D^2) > 0])
pr("correlação média entre as diferenças: %.2f",
   mean(Rm[upper.tri(Rm)]))
# O vencedor: otimismo da validação
w <- which.max(acv)
pr("vencedor %s: val %.4f fim %.4f (dif %.4f)", names(C)[w], acv[w],
   acf[w], acv[w] - acf[w])
pr("todos: dif média val - fim %.4f", mean(acv - acf))
# Sem nenhuma diferença real: o melhor de 20 em 300 imagens
g <- gerador(2015); pa <- 0.96; ep <- sqrt(pa * (1 - pa) / 300)
z <- apply(matriz_normal(g, 20000, 20), 1, max)
pr("máx. de 20 nulos independentes: +%.4f (%.2f erros-padrão)",
   mean(z) * ep, mean(z))
