# Cap. 12: Aveia sem a casela Victory/0,6: o que ainda é estimável
a <- read.csv(file.path("dados", "yates_oats.csv"))
a$gen <- factor(a$gen); a$dose <- factor(a$nitro)
b <- a[!(a$gen == "Victory" & a$nitro == 0.6), ]
cel <- interaction(b$gen, b$dose, drop = TRUE)  # 11 caselas
mc <- lm(yield ~ 0 + cel + block, data = b,     # médias + blocos
         contrasts = list(block = "contr.sum"))
s2 <- sum(resid(mc)^2) / mc$df.residual
c(n = nrow(b), posto = mc$rank, gl = mc$df.residual, s = sqrt(s2))
# Modelo de efeitos com interação: posto incompleto
Xe <- model.matrix(~ block + gen * dose, data = b,
                   contrasts.arg = list(block = "contr.sum"))
G <- MASS::ginv(crossprod(Xe)); bh <- G %*% crossprod(Xe, b$yield)
P <- diag(ncol(Xe)) - G %*% crossprod(Xe)    # l estimável <=> l'P = 0
estim <- function(l) max(abs(l %*% P))
# linha de lambda para cada média de casela: média das linhas do
# delineamento completo sobre os 6 blocos (soma zero zera os blocos)
Xa <- model.matrix(~ block + gen * dose, data = a,
                   contrasts.arg = list(block = "contr.sum"))
M <- rowsum(Xa, paste(a$gen, a$nitro)) / 6
linha <- function(g, d) M[paste(g, d), ]
dif <- function(ds) Reduce(`+`, lapply(ds, function(d)
  linha("Victory", d) - linha("GoldenRain", d))) / length(ds)
L <- rbind(doses_0_a_0.4 = dif(c("0", "0.2", "0.4")),
           todas_as_doses = dif(c("0", "0.2", "0.4", "0.6")),
           so_dose_0.6 = dif("0.6"))
res <- cbind(desvio_espaco_linha = apply(L, 1, estim),
             estimativa = drop(L %*% bh),
             ep = sqrt(s2 * diag(L %*% G %*% t(L))))
round(res, 3)
# Sob aditividade (sem interação), a média sobre as doses é estimável
ma <- lm(yield ~ block + gen + dose, data = b)
ea <- c(coef(ma)["genVictory"],
        sqrt(vcov(ma)["genVictory", "genVictory"]))
# Dados completos, modelo com interação (referência)
cf12 <- interaction(a$gen, a$dose)
mf <- lm(yield ~ 0 + cf12 + block, data = a,
         contrasts = list(block = "contr.sum"))
cf <- coef(mf); V <- vcov(mf)
lf <- (grepl("Victory", names(cf)) -
       grepl("GoldenRain", names(cf))) / 4
ef <- c(sum(lf * cf), sqrt(drop(t(lf) %*% V %*% lf)))
r <- rbind(aditivo_sem_casela = ea, completo_interacao = ef)
colnames(r) <- c("estimativa", "ep"); round(r, 3)
