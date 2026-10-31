# Cap. 2: Conhecendo os dados de Longley: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
l <- read.csv(file.path("dados", "longley.csv"))
v <- c("Employed", "GNP.deflator", "GNP", "Unemployed",
       "Armed.Forces", "Population", "Year")
resumo(l, v)
correlacoes(l, v)
