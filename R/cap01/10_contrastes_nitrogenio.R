# Cap. 1 -- Decomposição da SQ de tratamentos em contrastes polinomiais ortogonais
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
y <- aveia$yield
dose <- factor(aveia$nitro)
medias <- tapply(y, dose, mean)
r <- 18                                          # parcelas por dose

# Coeficientes linear, quadrático e cúbico (Gram-Schmidt em 1, t, t^2, t^3)
C <- cbind(linear = c(-3, -1, 1, 3), quadratico = c(1, -1, -1, 1), cubico = c(-1, 3, -3, 1))
crossprod(C)                                     # ortogonais entre si e a (1,1,1,1)

# Cada contraste c define a direção sum_j c_j d_j em R^72; a projeção de y sobre ela
# tem comprimento ao quadrado r (c'm)^2 / c'c
sq <- apply(C, 2, function(c) r * sum(c * medias)^2 / sum(c^2))
round(cbind(SQ = sq, fracao = sq / sum(sq)), 4)
sum(sq)                                          # = SQ de tratamentos (20020.5)
