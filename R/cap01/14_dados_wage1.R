# Cap. 1 -- Conhecendo os dados wage1: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
w <- read.csv(file.path("dados", "wage1.csv"))
w$lwage <- log(w$wage)                  # log do salário-hora
resumo(w, c("wage", "lwage", "educ", "exper", "tenure", "female"))
correlacoes(w, c("lwage", "educ", "exper", "tenure"))
# Log do salário por faixa de escolaridade
faixa <- cut(w$educ, c(-1, 11, 12, 15, 18),
             labels = c("a:0-11", "b:12", "c:13-15", "d:16-18"))
por_grupo(w$lwage, as.character(faixa), "escolaridade")
