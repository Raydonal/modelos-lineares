# Cap. 1 -- Conhecendo os dígitos manuscritos
d <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(d[, -1]); dim(X)          # 1797 imagens, 64 pixels
table(d$digito)                               # classes equilibradas
c(media_pixel = mean(X), frac_zeros = mean(X == 0))
# Tinta por imagem (soma das intensidades) em cada dígito
tinta <- rowSums(X)
round(tapply(tinta, d$digito, mean))
# Pixels que nunca acendem e pixels mais variáveis
sum(apply(X, 2, max) == 0)                    # sempre zero
v <- apply(X, 2, var)
names(sort(v, decreasing = TRUE))[1:5]
