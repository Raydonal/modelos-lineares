# Cap. 11 -- Aveia: parametrizações do modelo com interação
a <- read.csv(file.path("dados", "yates_oats.csv"))
a$gen <- factor(a$gen); a$dose <- factor(a$nitro)
mc <- lm(yield ~ 0 + gen:dose, data = a)      # médias das caselas
mt <- lm(yield ~ gen * dose, data = a)        # tratamento (padrão)
ms <- lm(yield ~ gen * dose, data = a,        # soma zero
         contrasts = list(gen = "contr.sum", dose = "contr.sum"))
mp <- lm(yield ~ gen * dose, data = a,        # Helmert e polinomial
         contrasts = list(gen = "contr.helmert", dose = "contr.poly"))
f <- sapply(list(mc, mt, ms, mp), fitted)
c(max_dif_ajustados = max(abs(f - f[, 1])))
mu <- matrix(coef(mc), 3, 4,
             dimnames = list(levels(a$gen), levels(a$dose)))
round(mu, 2)
# O "efeito de Marvellous" em duas parametrizações
round(c(tratamento = coef(mt)[["genMarvellous"]],
  dif_dose0 = mu["Marvellous", "0"] - mu["GoldenRain", "0"],
  soma = coef(ms)[["gen2"]],
  dif_media = mean(mu["Marvellous", ]) - mean(mu)), 3)
# Uma casela inteira perdida: Victory na dose 0,6
b <- a[!(a$gen == "Victory" & a$nitro == 0.6), ]
cel <- interaction(b$gen, b$dose, drop = TRUE)
mc2 <- lm(b$yield ~ 0 + cel)                  # 11 médias
mt2 <- lm(yield ~ gen * dose, data = b)
c(medias = length(coef(mc2)), colunas_efeitos = length(coef(mt2)),
  posto = mt2$rank, na = sum(is.na(coef(mt2))))
# Victory - GoldenRain nas doses 0, 0,2 e 0,4: estimável
m2 <- coef(mc2)
v <- function(g, d) m2[[paste0("cel", g, ".", d)]]
round(mean(sapply(c("0", "0.2", "0.4"), function(d)
  v("Victory", d) - v("GoldenRain", d))), 3)
