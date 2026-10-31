# Cap. 1 -- Conhecendo o experimento de aveia
source(file.path("R", "comum", "descritiva.R"))
a <- read.csv(file.path("dados", "yates_oats.csv"))
resumo(a, "yield")
table(variedade = a$gen, dose = a$nitro)    # 6 parcelas por casela
por_grupo(a$yield, a$gen, "variedade")
por_grupo(a$yield, a$nitro, "dose")
por_grupo(a$yield, a$block, "bloco")
# Médias das caselas variedade x dose
round(tapply(a$yield, list(a$gen, a$nitro), mean), 1)
