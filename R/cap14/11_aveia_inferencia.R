# Cap. 14: Aveia: aditividade, forma da dose e Dunnett
source(file.path("R", "comum", "multiplas.R"))
pr <- function(...) cat(sprintf(...), "\n", sep = "")
a <- read.csv(file.path("dados", "yates_oats.csv"))
a <- a[order(a$block, a$gen, a$nitro), ]
y <- a$yield; n <- length(y)
um <- function(f) {
  f <- as.character(f); outer(f, sort(unique(f)), "==") * 1
}
Bl <- um(a$block); Gn <- um(a$gen); Do <- um(a$nitro)
Pc <- um(paste(a$block, a$gen)); Ce <- um(paste(a$gen, a$nitro))
sqr <- function(...) {
  Z <- cbind(...); qz <- qr(Z)
  sum(qr.resid(qz, y)^2)
}
# Estrato das subparcelas: parcelas + doses + variedade x dose
s_pd <- sqr(Pc, Do); s_int <- sqr(Pc, Ce)
s_p <- sqr(Pc)
qmsub <- s_int / 45
pr("SQ erro sub %.2f QM %.2f gl 45", s_int, qmsub)
Fi <- ((s_pd - s_int) / 6) / qmsub
pr("interação: SQ %.1f F(6,45) %.3f p %.3f", s_pd - s_int, Fi,
   pf(Fi, 6, 45, lower.tail = FALSE))
Fd <- ((s_p - s_pd) / 3) / qmsub
pr("dose: SQ %.1f F(3,45) %.2f p %.1e", s_p - s_pd, Fd,
   pf(Fd, 3, 45, lower.tail = FALSE))
# Mesma hipótese com o erro ingênuo (72 - 17 = 55 gl)
s_b <- sqr(Bl, Gn, Do); s_c <- sqr(Bl, Ce)
Fn <- ((s_b - s_c) / 6) / (s_c / 55)
pr("interação ingênua: QM %.1f F(6,55) %.3f p %.3f", s_c / 55, Fn,
   pf(Fn, 6, 55, lower.tail = FALSE))
# Forma da dose dentro das parcelas: reta contra as 4 médias
d <- a$nitro; s_lin <- sqr(Pc, d)
Fl <- ((s_lin - s_pd) / 2) / qmsub
pr("reta: SQ lin %.1f falta %.1f F(2,45) %.3f p %.3f",
   s_p - s_lin, s_lin - s_pd, Fl, pf(Fl, 2, 45, lower.tail = FALSE))
s_q <- sqr(Pc, d, d^2)
Fq <- ((s_q - s_pd) / 1) / qmsub
pr("parábola: falta %.1f F(1,45) %.3f p %.3f", s_q - s_pd, Fq,
   pf(Fq, 1, 45, lower.tail = FALSE))
Fc <- ((s_lin - s_q) / 1) / qmsub
pr("termo quadrático: SQ %.1f F(1,45) %.3f p %.3f", s_lin - s_q, Fc,
   pf(Fc, 1, 45, lower.tail = FALSE))
# Tukey (um grau) nas médias de parcela: bloco x variedade
Y <- tapply(y, list(a$block, a$gen), mean)
A <- rowMeans(Y) - mean(Y); B <- colMeans(Y) - mean(Y)
E <- Y - outer(A, B, "+") - mean(Y)
D <- sum(outer(A, B) * Y)^2 / (sum(A^2) * sum(B^2))
gl <- 5 * 2 - 1
Ft <- D / ((sum(E^2) - D) / gl)
pr("Tukey: SQRes %.1f D %.1f F(1,9) %.3f p %.3f", sum(E^2), D, Ft,
   pf(Ft, 1, gl, lower.tail = FALSE))
Fg <- (4 * 6 * sum(B^2) / 2) / (4 * sum(E^2) / 10)
pr("variedade: F(2,10) %.3f p %.3f (QM parcelas %.1f)", Fg,
   pf(Fg, 2, 10, lower.tail = FALSE), 4 * sum(E^2) / 10)
# Doses contra a dose zero e todos os pares (r = 18)
md <- tapply(y, a$nitro, mean); ep <- sqrt(2 * qmsub / 18)
pr("médias %.2f %.2f %.2f %.2f ep %.3f", md[1], md[2], md[3], md[4],
   ep)
mt <- qt(0.975, 45); mb <- qt(1 - 0.05 / 6, 45)
md3 <- q_dunnett(3, 45); mtk <- qtukey(0.95, 4, 45) / sqrt(2)
msc <- sqrt(3 * qf(0.95, 3, 45))
pr("mult.: t %.3f Bonf %.3f Dunn %.3f Tukey %.3f Sch %.3f",
   mt, mb, md3, mtk, msc)
for (k in 2:4) pr("%.1f - 0: %.2f +- %.2f (Dunnett)",
  as.numeric(names(md)[k]), md[k] - md[1], md3 * ep)
dfp <- diff(md)
pr("pares vizinhos %.2f %.2f %.2f, Tukey +- %.2f", dfp[1], dfp[2],
   dfp[3], mtk * ep)
