# Cap. 9 -- Melhor preditor linear, precisão e correlações parciais
w <- read.csv(file.path("dados", "wage1.csv"))
D <- data.frame(lsal = log(w$wage), educ = w$educ, exper = w$exper,
                perman = w$tenure)
S <- cov(D); m <- colMeans(D)

# Melhor preditor linear de lsal: b = S_xx^{-1} S_xy (= MQ)
b <- solve(S[-1, -1], S[-1, 1])
rbind(momentos = c(m[1] - sum(b * m[-1]), b),
      mq = coef(lm(lsal ~ educ + exper + perman, data = D)))
c(R2 = drop(S[1, -1] %*% b) / S[1, 1],
  R2_lm = summary(lm(lsal ~ ., data = D))$r.squared)

# Correlações parciais por K = S^{-1}: -k_ij / sqrt(k_ii k_jj)
K <- solve(S)
P <- -K / sqrt(diag(K) %o% diag(K)); diag(P) <- 1
round(cor(D), 3)
round(P, 3)
# Conferência: resíduos de lsal e educ após (exper, perman)
r1 <- resid(lm(lsal ~ exper + perman, data = D))
r2 <- resid(lm(educ ~ exper + perman, data = D))
cor(r1, r2)
