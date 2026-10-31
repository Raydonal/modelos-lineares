# Cap. 9: Conhecendo os índices europeus: retornos diários
source(file.path("R", "comum", "descritiva.R"))
eu <- read.csv(file.path("dados", "eustock.csv"))
r <- as.data.frame(100 * diff(log(as.matrix(eu[, -1]))))  # em %
resumo(r)
correlacoes(r)
# Caudas pesadas: curtose (3 na normal) e dias além de 4 dp
curt <- sapply(r, function(x) mean((x - mean(x))^4) / mean(
  (x - mean(x))^2)^2)
alem4 <- sapply(r, function(x) sum(abs(x - mean(x)) > 4 * sd(x)))
round(rbind(curtose = curt, dias_alem_4dp = alem4), 2)
