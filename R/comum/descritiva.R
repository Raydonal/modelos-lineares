# Estatísticas descritivas com a mesma formatação em R e em Python
# (python/comum/descritiva.py): uma linha por variável, <= 70 colunas.
.cab <- c("média", "dp", "mín", "Q1", "mediana", "Q3", "máx")
# casas decimais pela ordem de grandeza da linha (igual no Python)
.dec <- function(s) { m <- max(abs(s))
  if (m >= 1000) 0 else if (m >= 100) 1 else if (m >= 10) 2 else 3 }
.fmt <- function(s, w) sprintf(paste0("%", w, ".", .dec(s), "f"), s)

resumo <- function(d, vars = names(d)) {
  cat(formatC("", width = -9), formatC("n", width = 5),
      paste(formatC(.cab, width = 8), collapse = ""), "\n", sep = "")
  for (v in vars) {
    x <- d[[v]]; x <- x[!is.na(x)]
    q <- quantile(x, c(0, 0.25, 0.5, 0.75, 1), names = FALSE)
    s <- c(mean(x), sd(x), q[1], q[2], q[3], q[4], q[5])
    cat(formatC(substr(v, 1, 9), width = -9),
        sprintf("%5d", length(x)),
        paste(.fmt(s, 8), collapse = ""), "\n", sep = "")
  }
  invisible(NULL)
}

correlacoes <- function(d, vars = names(d)) {
  r <- cor(d[vars], use = "pairwise.complete.obs")
  nm <- substr(vars, 1, 7)
  cat(formatC("", width = -9),
      paste(formatC(nm, width = 8), collapse = ""), "\n", sep = "")
  for (i in seq_along(vars))
    cat(formatC(substr(vars[i], 1, 9), width = -9),
        paste(sprintf("%8.3f", r[i, ]), collapse = ""), "\n",
        sep = "")
  invisible(NULL)
}

# Médias de y por grupos (um fator), com n e desvio-padrão
por_grupo <- function(y, g, rotulo = "grupo") {
  niv <- sort(unique(g))
  cat(formatC(rotulo, width = -12), formatC("n", width = 5),
      paste(formatC(c("média", "dp", "mín", "máx"), width = 9),
            collapse = ""), "\n", sep = "")
  for (k in niv) {
    x <- y[g == k & !is.na(y)]
    cat(formatC(substr(as.character(k), 1, 12), width = -12),
        sprintf("%5d", length(x)),
        paste(.fmt(c(mean(x), sd(x), min(x), max(x)), 9),
              collapse = ""), "\n", sep = "")
  }
  invisible(NULL)
}
