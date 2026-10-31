# Cap. 4 -- Conhecendo a temporada de 1981 da NFL
source(file.path("R", "comum", "descritiva.R"))
j <- read.csv(file.path("dados", "nfl.csv")); j <- j[j$ano == 1981, ]
j$margem <- j$pontos_mandante - j$pontos_visitante
resumo(j, c("pontos_mandante", "pontos_visitante", "margem", "linha"))
c(vitorias_mandante = mean(j$margem > 0),
  empates = sum(j$margem == 0),
  cor_margem_linha = cor(j$margem, j$linha))
# Saldo de pontos (marcados - sofridos): 3 melhores e 3 piores
saldo <- tapply(c(j$margem, -j$margem),
                c(j$mandante, j$visitante), sum)
s <- sort(saldo, decreasing = TRUE); c(head(s, 3), tail(s, 3))
