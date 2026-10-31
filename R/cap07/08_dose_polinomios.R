# Cap. 7: Resposta da aveia ao nitrogênio: condicionamento de bases polinomiais
a <- read.csv(file.path("dados", "yates_oats.csv"))
dose <- a$nitro * 50.80235 / 0.4046856                      # cwt/acre -> kg de N por hectare
y <- a$yield; n <- length(y)
sort(unique(round(dose, 1)))
kappa2 <- function(M) { s <- svd(M)$d; max(s) / min(s) }
bases <- list(monomios = outer(dose, 0:3, `^`),
              centrados = outer(dose - mean(dose), 0:3, `^`),
              padronizados = outer((dose - mean(dose)) / sd(dose), 0:3, `^`),
              ortogonais = cbind(1 / sqrt(n), poly(dose, 3)))   # colunas ortonormais
k <- sapply(bases, kappa2)
rbind(kappa_X = signif(k, 4), kappa_XtX = signif(k^2, 4))
# Os valores ajustados são os mesmos (C(X) é o mesmo); os coeficientes e sua estabilidade, não
aj <- sapply(bases, function(B) qr.fitted(qr(B), y))
max(abs(aj - aj[, 1]))
# Erro das equações normais em relação à QR, em cada base
sapply(bases, function(B) { b1 <- solve(crossprod(B), crossprod(B, y)); b2 <- qr.coef(qr(B), y)
  signif(max(abs(b1 - b2) / abs(b2)), 3) })
# Maior correlação (em módulo) entre as estimativas dos coeficientes: cov2cor((X'X)^{-1})
sapply(bases, function(B) { R <- cov2cor(solve(crossprod(B))); round(max(abs(R[upper.tri(R)])), 4) })
# Coeficientes na base ortogonal: um por grau (linear, quadrático, cúbico)
round(qr.coef(qr(bases$ortogonais), y), 3)
