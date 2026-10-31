# Cap. 3: Estática comparativa no modelo IS-LM pela regra de Cramer
# IS: Y = c0 + c1 (Y - T) + i0 - b r + G      LM: M/P = k Y - h r
par <- list(c0 = 100, c1 = 0.75, i0 = 1050, b = 50, k = 0.25, h = 100,
            T = 200, G = 250, MP = 500)
sistema <- function(p) {
  A <- with(p, matrix(c(1 - c1, b,
                        k,     -h), 2, byrow = TRUE))      # incógnitas (Y, r)
  d <- with(p, c(c0 - c1 * T + i0 + G, MP))
  list(A = A, d = d)
}
s <- sistema(par)
det(s$A)                                                    # -(h(1-c1) + b k) < 0
cramer <- function(A, d) sapply(1:2, function(j) { Aj <- A; Aj[, j] <- d; det(Aj) / det(A) })
eq <- cramer(s$A, s$d); names(eq) <- c("Y", "r"); eq

# Multiplicadores: derivadas do equilíbrio em relação a G e a M/P (Cramer com d = e_j)
mult <- cbind(dG = cramer(s$A, c(1, 0)), dMP = cramer(s$A, c(0, 1)))
rownames(mult) <- c("Y", "r"); mult
with(par, c(dY_dG_formula = h / (h * (1 - c1) + b * k)))   # forma fechada
p2 <- par; p2$G <- par$G + 1                                 # conferência numérica
cramer(sistema(p2)$A, sistema(p2)$d) - eq
