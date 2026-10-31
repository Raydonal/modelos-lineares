# Cap. 4 -- Restrições de identificação no experimento de aveia e inversas generalizadas
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
ind <- function(f) model.matrix(~ factor(f) - 1)
X <- cbind(1, ind(aveia$block), ind(aveia$nitro), ind(aveia$gen))    # 72 x 14, posto 11
colnames(X) <- c("mu", paste0("b", 1:6), paste0("n", c(0, 2, 4, 6)), c("GR", "Ma", "Vi"))
y <- aveia$yield
XtX <- crossprod(X); Xty <- crossprod(X, y)
e <- function(j) as.numeric(colnames(X) %in% j)           # vetor com uns nas posições j

# Três conjuntos de restrições K b = 0, cada um com 14 - 11 = 3 linhas
K <- list(
  tratamento = rbind(e("b1"), e("n0"), e("GR")),          # primeiro nível de cada fator = 0
  soma_zero  = rbind(e(paste0("b", 1:6)), e(paste0("n", c(0, 2, 4, 6))), e(c("GR", "Ma", "Vi"))),
  internas   = rbind(e("mu") - e(paste0("b", 1:6)), e("mu") - e(paste0("n", c(0, 2, 4, 6))),
                     e("mu") - e(c("GR", "Ma", "Vi"))))   # linhas de uma base de N(X)
sol <- sapply(K, function(Kk) {
  stopifnot(qr(rbind(X, Kk))$rank == 14)                  # [X; K] tem posto coluna completo
  G <- solve(XtX + crossprod(Kk))                         # é inversa generalizada de X'X
  stopifnot(max(abs(XtX %*% G %*% XtX - XtX)) < 1e-8)
  b <- G %*% Xty
  stopifnot(max(abs(Kk %*% b)) < 1e-8)                    # satisfaz as restrições
  drop(b)
})
sol <- cbind(sol, ginv = drop(MASS::ginv(XtX) %*% Xty))   # Moore-Penrose
round(sol[c("mu", "b1", "n0", "n6", "GR", "Vi"), ], 2)

# Funções lambda' b: estimável se lambda' G X'X = lambda' (lambda no espaço linha de X)
G <- MASS::ginv(XtX)
estimavel <- function(l) max(abs(l %*% G %*% XtX - l)) < 1e-8
L <- rbind("n6 - n0"           = e("n6") - e("n0"),
           "Vi - GR"           = e("Vi") - e("GR"),
           "mu + b1 + n0 + GR" = e(c("mu", "b1", "n0", "GR")),
           "mu"                = e("mu"),
           "n6"                = e("n6"))
data.frame(round(L %*% sol, 3), estimavel = apply(L, 1, estimavel), check.names = FALSE)
max(abs(X %*% (sol - sol[, 1])))                          # valores ajustados idênticos
