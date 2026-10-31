# Cap. 1: Conhecendo os dados UKgas: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
g <- read.csv(file.path("dados", "ukgas.csv"))
g$log_cons <- log(g$consumo)
resumo(g, c("consumo", "log_cons"))
por_grupo(g$consumo, paste0("tri", g$trimestre), "consumo")
por_grupo(g$log_cons, paste0("tri", g$trimestre), "log(consumo)")
# Amplitude sazonal (máx - mín dentro do ano), média por década
amp <- function(x) tapply(x, g$ano, function(v) diff(range(v)))
dec <- paste0(10 * (unique(g$ano) %/% 10), "s")
round(rbind(nivel = tapply(amp(g$consumo), dec, mean),
            log = tapply(amp(g$log_cons), dec, mean)), 2)
