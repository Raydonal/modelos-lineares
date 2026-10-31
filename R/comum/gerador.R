# Gerador pseudoaleatório portátil (Park-Miller, a = 48271, m = 2^31 - 1) com
# Box-Muller: produz exatamente os mesmos números em R e em Python
# (python/comum/gerador.py), para exemplos sintéticos cujos números são citados no texto.
gerador <- function(semente) {
  estado <- semente
  unif <- function(k) {
    u <- numeric(k)
    for (i in seq_len(k)) { estado <<- (48271 * estado) %% 2147483647; u[i] <- estado / 2147483647 }
    u
  }
  normal <- function(k) {
    u <- matrix(unif(2 * k), 2)
    sqrt(-2 * log(u[1, ])) * cos(2 * base::pi * u[2, ])
  }
  list(unif = unif, normal = normal)
}
# matriz k1 x k2 preenchida por linhas (igual ao reshape padrão do NumPy)
matriz_normal <- function(g, k1, k2, sd = 1) matrix(sd * g$normal(k1 * k2), k1, k2, byrow = TRUE)
