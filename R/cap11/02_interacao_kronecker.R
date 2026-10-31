# Cap. 11: Interações como produtos de Kronecker linha a linha
a <- read.csv(file.path("dados", "yates_oats.csv"))
ga <- as.integer(factor(a$gen)); nb <- as.integer(factor(a$nitro))
ind <- function(f, k) outer(f, 1:k, "==") * 1  # indicadoras
DA <- ind(ga, 3); DB <- ind(nb, 4)
DAB <- ind((ga - 1) * 4 + nb, 12)             # caselas, e_a (x) e_b
TA <- cbind(1, contr.treatment(3)); TB <- cbind(1, contr.treatment(4))
FA <- DA %*% TA; FB <- DB %*% TB              # [1 C] por observação
X <- matrix(0, nrow(a), 12)                   # produtos linha a linha
for (i in 1:3) for (j in 1:4)
  X[, (i - 1) * 4 + j] <- FA[, i] * FB[, j]
c(dif_kronecker = max(abs(X - DAB %*% kronecker(TA, TB))),
  posto = qr(X)$rank)
P <- function(Z) Z %*% MASS::ginv(Z)          # projeção sobre C(Z)
max(abs(P(X) - P(DAB)))                       # mesmo espaço coluna
# Perder uma casela inteira (Victory, dose 0,6): 6 parcelas
fora <- a$gen == "Victory" & a$nitro == 0.6
c(colunas = ncol(X), posto = qr(X[!fora, ])$rank,
  caselas_cheias = sum(colSums(DAB[!fora, ]) > 0))
