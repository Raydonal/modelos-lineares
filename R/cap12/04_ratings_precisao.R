# Cap. 12: Ratings de Massey: estimabilidade e precisão
nfl <- read.csv(file.path("dados", "nfl.csv"))
s <- subset(nfl, ano == 1981)
times <- sort(unique(c(s$mandante, s$visitante)))
delineamento <- function(j) {         # +1 mandante, -1 visitante
  X <- matrix(0, nrow(j), length(times), dimnames = list(NULL, times))
  X[cbind(seq_len(nrow(j)), match(j$mandante, times))] <- 1
  X[cbind(seq_len(nrow(j)), match(j$visitante, times))] <- -1
  cbind(X, mando = 1)
}
lam <- function(i, j) { l <- setNames(numeric(29), c(times, "mando"))
  l[i] <- 1; l[j] <- -1; l }
ajuste <- function(j) {
  X <- delineamento(j); y <- j$pontos_mandante - j$pontos_visitante
  G <- MASS::ginv(crossprod(X)); b <- drop(G %*% crossprod(X, y))
  r <- qr(X)$rank; s2 <- sum((y - X %*% b)^2) / (nrow(X) - r)
  list(X = X, G = G, b = b, s2 = s2, gl = nrow(X) - r)
}
f <- ajuste(s); l <- lam("SF", "DAL")
ep <- sqrt(f$s2 * drop(t(l) %*% f$G %*% l))
round(c(s = sqrt(f$s2), gl = f$gl, SF_DAL = sum(l * f$b), ep = ep,
        t = sum(l * f$b) / ep), 3)
# Precisão de todas as 378 diferenças e da diferença 1o - 2o
D <- outer(1:28, 1:28, Vectorize(function(i, j)
  if (i < j) drop(t(lam(times[i], times[j])) %*% f$G %*%
                  lam(times[i], times[j])) else NA))
round(c(ep_min = sqrt(f$s2 * min(D, na.rm = TRUE)),
        ep_medio = mean(sqrt(f$s2 * D), na.rm = TRUE),
        ep_max = sqrt(f$s2 * max(D, na.rm = TRUE))), 2)
l2 <- lam("PHA", "SF")
round(c(PHA_SF = sum(l2 * f$b),
        ep = sqrt(f$s2 * drop(t(l2) %*% f$G %*% l2))), 2)
# Semana a semana: SF - DAL é estimável? Com que variância (em s2)?
t(sapply(2:16, function(w) {
  fw <- ajuste(subset(s, semana <= w)); X <- fw$X
  P <- diag(29) - fw$G %*% crossprod(X)  # l estimável <=> l'P = 0
  e <- max(abs(l %*% P)) < 1e-8
  c(semana = w, jogos = nrow(X), estimavel = e,
    var_s2 = if (e) round(drop(t(l) %*% fw$G %*% l), 3) else NA)
}))
