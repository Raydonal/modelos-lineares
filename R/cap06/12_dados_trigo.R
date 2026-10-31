# Cap. 6 -- Conhecendo o ensaio de trigo: 18 cultivares, 9 locais
source(file.path("R", "comum", "descritiva.R"))
d <- read.csv(file.path("dados", "yan_trigo.csv"))
resumo(d, "producao")
por_grupo(d$producao, d$ambiente, "ambiente")
Y <- tapply(d$producao, list(d$genotipo, d$ambiente), mean)
g <- sort(rowMeans(Y), decreasing = TRUE)
round(c(head(g, 3), tail(g, 3)), 2)            # melhores e piores
# Quanto da variação é de ambiente, de genótipo e de interação?
m <- mean(Y); A <- colMeans(Y) - m; G <- rowMeans(Y) - m
I <- Y - m - outer(G, A, "+")
sq <- c(ambiente = 18 * sum(A^2), genotipo = 9 * sum(G^2),
        interacao = sum(I^2))
round(100 * sq / sum(sq), 1)                   # em % da SQ total
# A melhor cultivar muda de ambiente para ambiente
rownames(Y)[apply(Y, 2, which.max)]
