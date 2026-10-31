# Cap. 11: Sazonalidade: indicadoras de mês ou harmônicos de Fourier
s <- read.csv(file.path("dados", "seatbelts.csv"))
y <- log(s$motoristas); n <- length(y); t <- 1:n
D <- outer(s$mes, 1:12, "==") * 1             # 12 indicadoras
fourier <- function(K) {                      # K harmônicos (K <= 6)
  F <- NULL
  for (k in seq_len(K)) {
    F <- cbind(F, cos(2 * pi * k * t / 12))
    if (k < 6) F <- cbind(F, sin(2 * pi * k * t / 12))
  }
  F
}
F6 <- fourier(6)                              # 11 colunas
G <- crossprod(F6)                            # F'F diagonal
c(fora_diag = max(abs(G - diag(diag(G)))), diag_min = min(diag(G)),
  diag_max = max(diag(G)))
Xd <- cbind(D, t, s$lei)                      # indicadoras
Xf <- cbind(1, F6, t, s$lei)                  # Fourier completo
fit <- function(X) X %*% qr.solve(X, y)
c(dif_ajustados = max(abs(fit(Xd) - fit(Xf))))
sqr <- sapply(0:6, function(K)
  sum((y - fit(cbind(1, fourier(K), t, s$lei)))^2))
round(rbind(K = 0:6, colunas = c(0, 2, 4, 6, 8, 10, 11),
            SQR = sqr, frac = (sqr[1] - sqr) / (sqr[1] - sqr[7])), 3)
lei <- sapply(c(2, 6), function(K) {          # efeito da lei
  X <- cbind(1, fourier(K), t, s$lei); qr.solve(X, y)[ncol(X)] })
round(c(K2 = lei[1], K6 = lei[2]), 4)
