# Cap. 7: Quanto os dados de Longley podem ser arredondados? (Beaton, Rubin e Barone, 1976)
L <- read.csv(file.path("dados", "longley.csv"))
X <- cbind(intercepto = 1, as.matrix(L[, 2:7])); y <- L$Employed
kappa2 <- function(M) { s <- svd(M)$d; max(s) / min(s) }
c(kappa_X = kappa2(X), kappa_XtX = kappa2(crossprod(X)),
  kappa_colunas_unitarias = kappa2(sweep(X, 2, sqrt(colSums(X^2)), "/")))
b <- qr.coef(qr(X), y); round(b, 4)
# Os dados publicados têm 0, 1 ou 3 casas decimais: perturbamos cada valor por um erro de
# arredondamento uniforme em (-meia unidade, +meia unidade) da última casa e reajustamos
casas <- c(GNP.deflator = 1, GNP = 3, Unemployed = 1, Armed.Forces = 1, Population = 3, Year = 0)
source(file.path("R", "comum", "gerador.R")); g <- gerador(76)
R <- t(replicate(1000, {
  Xp <- X; Xp[, -1] <- Xp[, -1] + matrix(g$unif(16 * 6) - 0.5, 16, 6) %*% diag(10^-casas)
  bp <- qr.coef(qr(Xp), y)
  c(rel_coef = sqrt(sum((bp - b)^2) / sum(b^2)), rel_dados = svd(Xp - X)$d[1] / svd(X)$d[1], bp)
}))
round(quantile(R[, "rel_coef"], c(0.05, 0.5, 0.95)), 3)        # variação relativa do vetor de coeficientes
signif(median(R[, "rel_dados"]), 3)                               # tamanho relativo da perturbação nos dados
signif(kappa2(X) * median(R[, "rel_dados"]), 3)                  # cota de primeira ordem kappa * ||dX||/||X||
# Variação relativa de cada coeficiente e fração de sorteios em que o sinal muda
round(rbind(mediana = apply(abs(sweep(R[, -(1:2)], 2, b)) / rep(abs(b), each = 1000), 2, median),
            muda_sinal = colMeans(sign(R[, -(1:2)]) != rep(sign(b), each = 1000))), 3)
