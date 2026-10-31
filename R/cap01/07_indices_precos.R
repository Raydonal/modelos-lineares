# Cap. 1: Índices de preços de Laspeyres, Paasche e Fisher como razões de produtos internos
cesta <- data.frame(
  bem = c("arroz", "feijão", "carne", "leite", "pão"),
  p0  = c(5.20, 8.90, 38.50, 4.60, 12.00),    # preços no período base (R$/unidade)
  p1  = c(6.10, 8.10, 44.90, 5.30, 13.20),    # preços no período atual
  q0  = c(10,   4,    6,     20,   8),        # quantidades no período base
  q1  = c(11,   5,    4.5,   19,   8)         # quantidades no período atual
)
ip <- function(p, q) sum(p * q)               # produto interno <p, q>

laspeyres <- ip(cesta$p1, cesta$q0) / ip(cesta$p0, cesta$q0)
paasche   <- ip(cesta$p1, cesta$q1) / ip(cesta$p0, cesta$q1)
fisher    <- sqrt(laspeyres * paasche)
round(c(Laspeyres = laspeyres, Paasche = paasche, Fisher = fisher), 4)

# Laspeyres como média dos relativos de preço com pesos de dispêndio
w <- cesta$p0 * cesta$q0 / ip(cesta$p0, cesta$q0)
data.frame(bem = cesta$bem, peso = round(w, 3), relativo = round(cesta$p1 / cesta$p0, 3))
sum(w * cesta$p1 / cesta$p0)                  # = Laspeyres
