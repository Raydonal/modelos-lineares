# Cap. 9 -- Conhecendo os dados airquality: faltantes e associações
source(file.path("R", "comum", "descritiva.R"))
a <- read.csv(file.path("dados", "airquality.csv"))
v <- c("Ozone", "Solar.R", "Wind", "Temp")
resumo(a, v)                                 # n conta os observados
colSums(is.na(a[v]))
correlacoes(a, v)                            # pares completos
# O ozônio falta mais em junho
round(tapply(is.na(a$Ozone), a$Month, mean), 2)
