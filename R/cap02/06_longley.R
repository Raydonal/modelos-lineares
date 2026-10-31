# Cap. 2 -- Mínimos quadrados nos dados de Longley: equações normais versus QR
lon <- read.csv(file.path("dados", "longley.csv"))
X <- cbind(1, as.matrix(lon[, c("GNP.deflator", "GNP", "Unemployed",
                                "Armed.Forces", "Population", "Year")]))
y <- lon$Employed

# Valores certificados pelo NIST (StRD), convertidos para as unidades do R
cert <- c(-3482.25863459582, 0.0150618722713733, -0.0358191792925910,
          -0.0202022980381683, -0.0103322686717359, -0.0511041056535807,
          1.82915146461355)

b_normais <- solve(crossprod(X), crossprod(X, y))  # resolve X'X b = X'y
f <- qr(X)
b_qr <- backsolve(qr.R(f), crossprod(qr.Q(f), y))  # resolve R b = Q'y

# LRE = número de algarismos corretos = -log10(erro relativo)
lre <- function(b) pmin(-log10(abs(b - cert) / abs(cert)), 15)
nomes <- c("(Intercepto)", colnames(X)[-1])
round(data.frame(normais = lre(as.vector(b_normais)), qr = lre(as.vector(b_qr)), row.names = nomes), 1)
kappa(X, exact = TRUE); kappa(crossprod(X), exact = TRUE)   # condição de X e de X'X
