# Exporta para dados/ (em CSV) os conjuntos de dados de pacotes R usados no
# livro, para que os scripts Python leiam exatamente os mesmos dados.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
raiz <- if (length(arg)) normalizePath(file.path(dirname(sub("^--file=", "", arg)), "..")) else "."
dir_dados <- file.path(raiz, "dados")
dir.create(dir_dados, showWarnings = FALSE)

exportar <- function(df, nome) {
  write.csv(df, file.path(dir_dados, paste0(nome, ".csv")), row.names = FALSE)
  message("dados/", nome, ".csv  (", nrow(df), " linhas)")
}

# Cap. 1 -------------------------------------------------------------------
exportar(agridat::yates.oats, "yates_oats")                       # agricultura
exportar(wooldridge::wage1[, c("wage", "educ", "exper", "tenure", "female",
                                "northcen", "south", "west")],
         "wage1")                                                  # economia
exportar(data.frame(ano = floor(time(UKgas)), trimestre = cycle(UKgas),
                    consumo = as.numeric(UKgas)), "ukgas")             # energia

# Cap. 2 -------------------------------------------------------------------
exportar(data.frame(ano = as.integer(rownames(longley)), longley, row.names = NULL),
         "longley")                                                # NIST StRD

# Cap. 4 -------------------------------------------------------------------
f <- LearnBayes::footballscores                     # NFL, temporadas 1981, 1983 e 1984
casa <- f$home == 1                                 # o favorito joga em casa?
exportar(data.frame(ano = f$year, semana = f$week,
                    mandante  = ifelse(casa, as.character(f$favorite.name), as.character(f$underdog.name)),
                    visitante = ifelse(casa, as.character(f$underdog.name), as.character(f$favorite.name)),
                    pontos_mandante  = ifelse(casa, f$favorite, f$underdog),
                    pontos_visitante = ifelse(casa, f$underdog, f$favorite),
                    linha = ifelse(casa, f$spread, -f$spread)),   # margem prevista p/ o mandante
         "nfl")                                                   # esportes

# Cap. 5 -------------------------------------------------------------------
sb <- datasets::Seatbelts                          # Reino Unido, mensal, 1969-1984
exportar(data.frame(ano = floor(time(sb) + 1e-8), mes = as.integer(cycle(sb)),
                    motoristas = as.numeric(sb[, "drivers"]),
                    km = as.numeric(sb[, "kms"]),
                    preco_gasolina = as.numeric(sb[, "PetrolPrice"]),
                    lei = as.numeric(sb[, "law"])),
         "seatbelts")                                              # transportes
m <- wooldridge::mroz
m <- m[m$inlf == 1, c("lwage", "educ", "exper", "expersq", "fatheduc", "motheduc")]
exportar(m, "mroz")                                                # economia (VI)
exportar(MASS::mcycle, "mcycle")                                  # aprendizado de máquina

# Cap. 6 -------------------------------------------------------------------
ww <- agridat::yan.winterwheat
exportar(data.frame(genotipo = ww$gen, ambiente = ww$env, producao = ww$yield), "yan_trigo")  # agricultura

# Cap. 1 (modelos de linguagem) e Cap. 6 (embeddings) ----------------------
jb <- janeaustenr::austen_books()
exportar(data.frame(livro = as.character(jb$book), texto = jb$text), "austen")   # linguagem

# Cap. 9 -------------------------------------------------------------------
eu <- datasets::EuStockMarkets                     # fechamento diário, 1991-1998
exportar(data.frame(dia = seq_len(nrow(eu)), as.data.frame(eu)), "eustock")   # finanças
exportar(datasets::airquality, "airquality")       # Nova York, mai.-set. de 1973 (clima)
