# Cap. 11 -- Codificações de um fator: o que cada coeficiente mede
a <- read.csv(file.path("dados", "yates_oats.csv"))
dose <- factor(a$nitro)                       # 4 níveis: 0 a 0,6
mu <- tapply(a$yield, dose, mean)             # médias das caselas
cods <- list(tratamento = contr.treatment(4), soma = contr.sum(4),
             helmert = contr.helmert(4), polinomial = contr.poly(4))
for (nome in names(cods)) {
  C <- cods[[nome]]
  M <- solve(cbind(1, C))                     # coeficientes = M mu
  dimnames(M) <- list(paste0("b", 0:3), paste0("mu", 1:4))
  X <- cbind(1, C[as.integer(dose), ])        # matriz do modelo
  b <- qr.solve(X, a$yield)
  cat(nome, ": max |b - M mu| =",
      signif(max(abs(b - M %*% mu)), 2), "\n")
  print(round(M, 3))
}
round(mu, 2)
