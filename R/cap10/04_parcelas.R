# Cap. 10: Parcelas subdivididas: dois estratos de erro (aveia)
av <- read.csv(file.path("dados", "yates_oats.csv"))
y <- av$yield; n <- length(y)
proj <- function(f) { D <- outer(f, unique(f), "==") * 1
  D %*% solve(crossprod(D)) %*% t(D) }
J <- matrix(1 / n, n, n)
PB <- proj(av$block); PV <- proj(av$gen); PN <- proj(av$nitro)
PW <- proj(paste(av$block, av$gen))           # parcelas principais
PNV <- proj(paste(av$nitro, av$gen))
A <- list(bloco = PB - J, variedade = PV - J,
          erro_a = PW - PB - PV + J,          # erro da parcela
          nitrogenio = PN - J, NxV = PNV - PN - PV + J,
          erro_b = diag(n) - PW - PNV + PV)   # erro da subparcela
# Ortogonalidade e a estrutura de Sigma = s2 I + sp2 D_W D_W'
DD <- 4 * PW                                  # D_W D_W' = 4 P_W
ort <- max(sapply(1:6, function(i) sapply(1:6, function(j)
  if (i == j) 0 else max(abs(A[[i]] %*% A[[j]])))))
estrato <- sapply(A, function(Ai)
  sum(diag(Ai %*% DD)) / sum(diag(Ai)))
c(max_AiAj = ort, round(estrato, 10))
gl <- sapply(A, function(Ai) round(sum(diag(Ai))))
SQ <- sapply(A, function(Ai) drop(t(y) %*% Ai %*% y)); QM <- SQ / gl
F <- c(QM[1:2] / QM["erro_a"], NA, QM[4:5] / QM["erro_b"], NA)
gl_den <- c(gl["erro_a"], gl["erro_a"], NA, gl["erro_b"],
            gl["erro_b"], NA)
data.frame(gl, SQ = round(SQ, 1), QM = round(QM, 1), F = round(F, 2),
           p = signif(pf(F, gl, gl_den, lower.tail = FALSE), 3))
s2 <- unname(QM["erro_b"]); sp2 <- unname((QM["erro_a"] - s2) / 4)
c(sigma2 = s2, sigma2_parcela = sp2)

# Poder para uma diferença de 10 entre o melhor e o pior nível
poder <- function(b) {
  tv <- c(-5, 0, 5); tn <- c(-5, -5 / 3, 5 / 3, 5)
  lv <- 4 * b * sum(tv^2) / (s2 + 4 * sp2)
  ln <- 3 * b * sum(tn^2) / s2
  c(blocos = b,
    variedade = pf(qf(0.95, 2, 2 * (b - 1)), 2, 2 * (b - 1), ncp = lv,
                   lower.tail = FALSE),
    nitrogenio = pf(qf(0.95, 3, 9 * (b - 1)), 3, 9 * (b - 1),
                    ncp = ln, lower.tail = FALSE))
}
round(t(sapply(c(3, 6, 9, 12), poder)), 3)
