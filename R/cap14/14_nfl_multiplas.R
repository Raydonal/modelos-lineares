# Cap. 14: Ratings da NFL: 378 comparações ao mesmo tempo
source(file.path("R", "comum", "gerador.R"))
source(file.path("R", "comum", "multiplas.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
nfl <- read.csv(file.path("dados", "nfl.csv"))
s <- subset(nfl, ano == 1981)
tm <- sort(unique(c(s$mandante, s$visitante))); k <- length(tm)
n <- nrow(s); y <- s$pontos_mandante - s$pontos_visitante
X <- matrix(0, n, k)
X[cbind(seq_len(n), match(s$mandante, tm))] <- 1
X[cbind(seq_len(n), match(s$visitante, tm))] <- -1
X <- cbind(X, 1)
G <- MASS::ginv(crossprod(X)); b <- drop(G %*% crossprod(X, y))
r <- qr(X)$rank; nu <- n - r
s2 <- sum((y - X %*% b)^2) / nu
pr("jogos %d posto %d gl %d s %.3f", n, r, nu, sqrt(s2))
pr("mando: %.2f ep %.2f", b[k + 1], sqrt(s2 * G[k + 1, k + 1]))
# Teste F global: todos os times iguais (q = k - 1)
Z <- X[, k + 1, drop = FALSE]
sq0 <- sum((y - Z %*% solve(crossprod(Z), crossprod(Z, y)))^2)
q <- k - 1; Fg <- ((sq0 - s2 * nu) / q) / s2
pr("F(%d,%d) %.3f p %.2e", q, nu, Fg,
   pf(Fg, q, nu, lower.tail = FALSE))
# Todas as diferenças
par <- which(upper.tri(diag(k)), arr.ind = TRUE)
par <- par[order(par[, 1], par[, 2]), ]; m <- nrow(par)
L <- matrix(0, m, k + 1)
L[cbind(seq_len(m), par[, 1])] <- 1
L[cbind(seq_len(m), par[, 2])] <- -1
est <- drop(L %*% b); LG <- L %*% G
se2 <- rowSums(LG * L); tt <- est / sqrt(s2 * se2)
pv <- 2 * pt(-abs(tt), nu)
# Valor crítico conjunto: máximo de |t| (lei simulada)
N <- 10000; g <- gerador(1981)
E <- matriz_normal(g, N, n)
ss <- sqrt(qchisq(g$unif(N), nu) / nu)
D <- (LG %*% t(X)) / sqrt(se2)
mx <- apply(abs(E %*% t(D)), 1, max) / ss
cmax <- sort(mx)[0.95 * N]
cs <- c(t = qt(0.975, nu), bonf = qt(1 - 0.025 / m, nu),
        maxt = cmax, sch = sqrt(q * qf(0.95, q, nu)))
pr("m %d t %.3f Bonf %.3f max-t %.3f Scheffé %.3f", m, cs[1], cs[2],
   cs[3], cs[4])
rb <- bh(pv, 0.05)
pr("pares: sem ajuste %d Bonf %d max-t %d Scheffé %d BH %d",
   sum(abs(tt) > cs[1]), sum(abs(tt) > cs[2]), sum(abs(tt) > cs[3]),
   sum(abs(tt) > cs[4]), rb)
pr("contraste máximo: |t| = sqrt(q F) = %.3f", sqrt(q * Fg))
pr("limiar BH: p <= %.5f, |t| >= %.3f", 0.05 * rb / m,
   qt(1 - 0.025 * rb / m, nu))
pr("sob ausência de efeito: esperadas %.1f falsas sem ajuste",
   0.05 * m)
o <- order(-b[1:k])
pr("melhores: %s", paste(tm[o[1:3]], collapse = " "))
pr("piores: %s", paste(tm[o[(k - 2):k]], collapse = " "))
pr("rating 1o - 2o %.2f; 1o - 28o %.2f", b[o[1]] - b[o[2]],
   b[o[1]] - b[o[k]])
top <- function(i) { l <- numeric(k + 1); l[o[1]] <- 1; l[o[i]] <- -1
  e <- sum(l * b); v <- sqrt(s2 * drop(t(l) %*% G %*% l))
  c(e, v, e / v) }
for (i in c(2, 5, 10, 14)) { v <- top(i)
  pr("1o - %2do: %.2f ep %.2f t %.2f", i, v[1], v[2], v[3]) }
pr("ep mínimo %.2f médio %.2f máximo %.2f", sqrt(s2 * min(se2)),
   mean(sqrt(s2 * se2)), sqrt(s2 * max(se2)))
