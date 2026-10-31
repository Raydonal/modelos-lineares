# Cap. 4 -- Ratings de Massey na NFL: o que é estimável num campeonato
nfl <- read.csv(file.path("dados", "nfl.csv"))
s <- subset(nfl, ano == 1981)
times <- sort(unique(c(s$mandante, s$visitante)))         # 28 times
delineamento <- function(j) {                              # +1 mandante, -1 visitante, 1 (mando)
  X <- matrix(0, nrow(j), length(times), dimnames = list(NULL, times))
  X[cbind(seq_len(nrow(j)), match(j$mandante, times))] <- 1
  X[cbind(seq_len(nrow(j)), match(j$visitante, times))] <- -1
  cbind(X, mando = 1)
}
X <- delineamento(s); y <- s$pontos_mandante - s$pontos_visitante
c(jogos = nrow(X), colunas = ncol(X), posto = qr(X)$rank, posto_times = qr(X[, 1:28])$rank)

# Três soluções das equações normais: Moore-Penrose, lm (último time = NA) e referência ATL = 0
b_mp  <- drop(MASS::ginv(crossprod(X)) %*% crossprod(X, y)); names(b_mp) <- colnames(X)
b_lm  <- coef(lm(y ~ X - 1)); b_lm[is.na(b_lm)] <- 0
K     <- c(1, rep(0, 28))                                  # restrição: rating do ATL = 0
b_ref <- drop(solve(crossprod(X) + tcrossprod(K), crossprod(X, y)))
sol <- cbind(mp = b_mp, lm = b_lm, ref = b_ref); rownames(sol) <- colnames(X)
round(sol[c("ATL", "DAL", "SF", "mando"), ], 2)
round(c(soma_ratings_mp = sum(b_mp[1:28]), media_margens = mean(y)), 4)

# Diferenças de ratings (e o mando) são estimáveis: iguais nas três soluções
dif <- function(b, i, j) b[i] - b[j]
round(rbind(SF_DAL = apply(sol, 2, dif, "SF", "DAL"), DAL_ATL = apply(sol, 2, dif, "DAL", "ATL")), 3)
head(round(sort(b_mp[1:28], decreasing = TRUE), 1), 5)

# Conexidade do calendário: número de componentes do grafo de jogos = 28 - posto
componentes <- function(j) {                               # rótulos por propagação do mínimo
  rot <- setNames(seq_along(times), times)
  repeat {
    novo <- rot
    for (k in seq_len(nrow(j))) novo[c(j$mandante[k], j$visitante[k])] <- min(novo[c(j$mandante[k], j$visitante[k])])
    if (all(novo == rot)) return(rot)
    rot <- novo
  }
}
sapply(1:6, function(w) {
  j <- subset(s, semana <= w)
  c(semana = w, componentes = length(unique(componentes(j))),
    "28 - posto" = 28 - qr(delineamento(j)[, 1:28])$rank)
})
split(times, componentes(subset(s, semana <= 2)))         # os dois grupos após a 2a semana

# Previsão: ratings com as semanas 1-8 preveem as semanas 9-16
tr <- subset(s, semana <= 8); te <- subset(s, semana > 8)
b8 <- MASS::ginv(delineamento(tr)) %*% (tr$pontos_mandante - tr$pontos_visitante)
m <- te$pontos_mandante - te$pontos_visitante; prev <- drop(delineamento(te) %*% b8)
round(c(EAM_massey = mean(abs(m - prev)), EAM_apostas = mean(abs(m - te$linha)),
        EAM_zero = mean(abs(m)), acerto_massey = mean(sign(prev) == sign(m)),
        acerto_apostas = mean(sign(te$linha) == sign(m))), 3)
