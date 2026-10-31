# Cap. 4: Existência e não unicidade de inversas generalizadas
A <- matrix(c(1, 2, 0,  1,
              2, 4, 1,  1,
              3, 6, 1,  2), 3, byrow = TRUE)    # 3 x 4, posto 2
qr(A)$rank

# (1) Pela fatoração de posto A = B C (colunas 1 e 3 formam uma base de C(A))
B <- A[, c(1, 3)]
C <- rbind(c(1, 2, 0, 1), c(0, 0, 1, -1))       # coeficientes das colunas na base
max(abs(B %*% C - A))
G1 <- t(C) %*% solve(C %*% t(C)) %*% solve(crossprod(B)) %*% t(B)

# (2) Por uma submatriz 2 x 2 não singular (linhas 1-2, colunas 1 e 3)
G2 <- matrix(0, 4, 3)
G2[c(1, 3), 1:2] <- solve(A[1:2, c(1, 3)])

# (3) Uma terceira, pela forma geral G = G0 + U - G0 A U A G0
set.seed(4)
U  <- matrix(round(rnorm(12), 1), 4, 3)
G3 <- G2 + U - G2 %*% A %*% U %*% A %*% G2

erro <- function(G) max(abs(A %*% G %*% A - A))  # verifica A G A = A
posto <- function(M) qr(M)$rank
Gs <- list(G1 = G1, G2 = G2, G3 = G3)
data.frame(erro_AGA = signif(sapply(Gs, erro), 2), posto_G = sapply(Gs, posto))
G2
round(G3, 2)
