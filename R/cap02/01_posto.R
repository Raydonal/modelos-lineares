# Cap. 2 -- Posto de matrizes e desigualdades de posto
posto <- function(A) qr(A)$rank                 # posto via QR (tolerância padrão)

A <- matrix(c(1, 2, 0, 3,
              2, 4, 1, 7,
              3, 6, 1, 10), nrow = 3, byrow = TRUE)   # linha 3 = linha 1 + linha 2
B <- matrix(c(1, 0,
              0, 1,
              1, 1,
              2, -1), nrow = 4, byrow = TRUE)
c(posto_A = posto(A), posto_At = posto(t(A)), posto_AtA = posto(crossprod(A)))
c(posto_AB = posto(A %*% B), cota = min(posto(A), posto(B)),
  sylvester = posto(A) + posto(B) - ncol(A))       # r(A) + r(B) - n <= r(AB)

# posto da soma: r(A + C) <= r(A) + r(C)
C <- outer(c(1, -1, 2), c(0, 1, 1, 0))            # matriz de posto 1
c(posto_soma = posto(A + C), cota = posto(A) + posto(C))
