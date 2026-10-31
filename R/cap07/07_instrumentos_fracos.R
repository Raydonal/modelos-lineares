# Cap. 7: Instrumentos fracos: a amplificação 1/|r| da projeção oblíqua (simulação)
source(file.path("R", "comum", "gerador.R")); g <- gerador(77)
n <- 428; beta <- 0.06                                     # tamanho e efeito como nos dados de Mroz
simula <- function(pi1) {
  z <- g$normal(n); u <- g$normal(n); v <- g$normal(n)
  x <- pi1 * z + 0.6 * u + 0.8 * v                         # regressor endógeno: depende de u
  y <- beta * x + 0.5 * u + 0.3 * g$normal(n)              # u também afeta y (confundimento)
  zc <- z - mean(z); xc <- x - mean(x); yc <- y - mean(y)
  r <- sum(zc * xc) / sqrt(sum(zc^2) * sum(xc^2))
  c(vi = sum(zc * yc) / sum(zc * xc), mq = sum(xc * yc) / sum(xc^2), r = r, F = r^2 * (n - 2) / (1 - r^2))
}
res <- sapply(c(0.4, 0.2, 0.1, 0.05), function(pi1) {
  s <- replicate(2000, simula(pi1))
  q <- quantile(s["vi", ], c(0.05, 0.25, 0.5, 0.75, 0.95))
  c(forca = pi1, r_mediano = median(s["r", ]), F_mediano = median(s["F", ]), vi_mediana = q[[3]],
    vi_iqr = q[[4]] - q[[2]], iqr_vezes_r = (q[[4]] - q[[2]]) * median(s["r", ]),
    vi_q05 = q[[1]], vi_q95 = q[[5]], mq_mediana = median(s["mq", ]))
})
round(res, 3)
