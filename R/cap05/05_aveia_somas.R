# Cap. 5 -- Somas de quadrados como projeções no experimento de aveia
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
proj <- function(X) { q <- qr(X); Q <- qr.Q(q)[, seq_len(q$rank), drop = FALSE]; Q %*% t(Q) }
ind  <- function(f) model.matrix(~ factor(f) - 1)

# Projeções sobre os efeitos centrados de cada fator: P_F = P_{C(indicadoras)} - J/n
pecas <- function(d) {
  n <- nrow(d); J <- matrix(1 / n, n, n)
  list(B = proj(ind(d$block)) - J, N = proj(ind(d$nitro)) - J, V = proj(ind(d$gen)) - J, J = J)
}
p <- pecas(aveia); y <- aveia$yield; n <- length(y)
comuta <- function(A, B) max(abs(A %*% B))
c(BN = comuta(p$B, p$N), BV = comuta(p$B, p$V), NV = comuta(p$N, p$V))
R <- diag(n) - p$J - p$B - p$N - p$V                    # resíduo
c(erro_idemp_R = max(abs(R %*% R - R)))
sq <- function(P, y) drop(t(y) %*% P %*% y)
data.frame(gl = sapply(list(p$B, p$N, p$V, R), function(M) round(sum(diag(M)))),
           SQ = sapply(list(p$B, p$N, p$V, R), sq, y = y), row.names = c("blocos", "nitrogênio", "variedade", "resíduo"))

# Seis subparcelas perdidas: as projeções deixam de ser ortogonais
source(file.path("R", "comum", "gerador.R"))             # mesmos sorteios que o Python
g <- gerador(5)
sorteia <- function() sort(order(g$unif(n))[1:6])        # 6 subparcelas ao acaso
perdidas <- sorteia(); perdidas
a2 <- aveia[-perdidas, ]; p2 <- pecas(a2)
c(BN = comuta(p2$B, p2$N), BV = comuta(p2$B, p2$V), NV = comuta(p2$N, p2$V))

# Soma de quadrados da variedade conforme a ordem de entrada no modelo
sq_var <- function(d) {
  y2 <- d$yield; um <- rep(1, nrow(d))
  B <- ind(d$block); N <- ind(d$nitro); V <- ind(d$gen)
  P0 <- proj(cbind(um)); PB <- proj(cbind(um, B)); PN <- proj(cbind(um, N)); PBN <- proj(cbind(um, B, N))
  Pc <- proj(cbind(um, B, N, V))
  sq_res <- sq(diag(nrow(d)) - Pc, y2); gl_res <- nrow(d) - 11
  ss <- c(V_primeiro = sq(proj(cbind(um, V)) - P0, y2), V_apos_B = sq(proj(cbind(um, B, V)) - PB, y2),
          V_apos_N = sq(proj(cbind(um, N, V)) - PN, y2), V_por_ultimo = sq(Pc - PBN, y2))
  p <- pf((ss[c(1, 4)] / 2) / (sq_res / gl_res), 2, gl_res, lower.tail = FALSE)
  c(ss, p_primeiro = p[[1]], p_ultimo = p[[2]])
}
round(sq_var(a2), 4)

# Repetição: 1000 sorteios de 6 subparcelas perdidas
R <- t(replicate(1000, sq_var(aveia[-sorteia(), ])))
razao <- R[, "V_por_ultimo"] / R[, "V_primeiro"]
round(quantile(razao, c(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)), 3)
c(muda_mais_de_20pc = mean(razao < 0.8 | razao > 1.25))
c(muda_conclusao = mean((R[, "p_primeiro"] < 0.05) != (R[, "p_ultimo"] < 0.05)),
  p_primeiro_signif = mean(R[, "p_primeiro"] < 0.05), p_ultimo_signif = mean(R[, "p_ultimo"] < 0.05))
