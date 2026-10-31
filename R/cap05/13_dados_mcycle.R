# Cap. 5: Conhecendo os dados mcycle: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
d <- read.csv(file.path("dados", "mcycle.csv"))
resumo(d)
c(instantes_distintos = length(unique(d$times)),
  repetidos = sum(duplicated(d$times)))
# A dispersão muda ao longo do tempo (heterocedasticidade)
fase <- cut(d$times, c(0, 14, 25, 40, 60),
            labels = c("a:antes", "b:queda", "c:oscila", "d:fim"))
por_grupo(d$accel, as.character(fase), "fase (ms)")
