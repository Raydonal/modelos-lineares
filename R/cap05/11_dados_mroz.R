# Cap. 5 -- Conhecendo os dados de Mroz: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
m <- read.csv(file.path("dados", "mroz.csv"))
v <- c("lwage", "educ", "exper", "fatheduc", "motheduc")
resumo(m, v)
correlacoes(m, v)
