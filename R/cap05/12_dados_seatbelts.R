# Cap. 5 -- Conhecendo os dados Seatbelts: estatísticas descritivas
source(file.path("R", "comum", "descritiva.R"))
s <- read.csv(file.path("dados", "seatbelts.csv"))
resumo(s, c("motoristas", "km", "preco_gasolina"))
# Antes e depois da lei (fevereiro de 1983)
por_grupo(s$motoristas, ifelse(s$lei == 1, "depois", "antes"),
          "lei")
# Perfil sazonal: média de mortos e feridos graves em cada mês
round(tapply(s$motoristas, s$mes, mean))
correlacoes(transform(s, lmot = log(motoristas), lkm = log(km),
                      lpreco = log(preco_gasolina)),
            c("lmot", "lkm", "lpreco", "lei"))
