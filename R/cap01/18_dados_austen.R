# Cap. 1 -- Conhecendo os romances de Jane Austen
d <- read.csv(file.path("dados", "austen.csv"))
pal <- function(t) {
  t <- tolower(t); unlist(regmatches(t, gregexpr("[a-z]+", t)))
}
por_livro <- t(sapply(split(d$texto, d$livro), function(t) {
  p <- pal(t); c(palavras = length(p), distintas = length(unique(p)))
}))
por_livro
p <- pal(d$texto); fr <- sort(table(p), decreasing = TRUE)
c(total = length(p), distintas = length(fr),
  uma_vez = sum(fr == 1))                     # hapax legomena
head(fr, 10)
# Lei de Zipf: frequência x posto é quase constante
k <- c(1, 10, 100, 1000, 10000)
data.frame(posto = k, freq = as.numeric(fr[k]),
           freq_x_posto = as.numeric(fr[k]) * k)
