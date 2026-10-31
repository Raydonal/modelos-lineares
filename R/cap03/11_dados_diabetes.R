# Cap. 3: Conhecendo os dados de diabetes: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
d <- read.csv(file.path("dados", "diabetes.csv"))
resumo(d)
# Correlação de cada covariável com a progressão y
round(sapply(d[1:10], function(x) cor(x, d$y)), 3)
# As medidas séricas s1 a s4 são fortemente correlacionadas
correlacoes(d, c("s1", "s2", "s3", "s4", "s5"))
