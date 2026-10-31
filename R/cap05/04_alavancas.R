# Cap. 5: A diagonal da matriz chapéu (alavancas) nos dados wage1
w <- read.csv(file.path("dados", "wage1.csv"))
X <- cbind(1, w$educ, w$exper, w$tenure); y <- log(w$wage)
n <- nrow(X); p <- ncol(X)
H <- X %*% solve(crossprod(X), t(X)); h <- diag(H)
c(soma = sum(h), p = p, min = min(h), um_sobre_n = 1 / n, max = max(h), media = mean(h))
max(abs(h - hatvalues(lm(y ~ educ + exper + tenure, data = w))))

# h_ii = 1/n + distância de Mahalanobis (ao quadrado) / (n - 1)
Zc <- scale(X[, -1], scale = FALSE)
S  <- crossprod(Zc) / (n - 1)
d2 <- rowSums((Zc %*% solve(S)) * Zc)
max(abs(h - (1 / n + d2 / (n - 1))))

# h_ii (1 - h_ii) = soma_{j != i} h_ij^2 e a derivada de yhat_i em relação a y_i
i <- which.max(h)
c(i = i, h_ii = h[i], lado_esq = h[i] * (1 - h[i]), lado_dir = sum(H[i, -i]^2))
w[i, c("wage", "educ", "exper", "tenure")]
y2 <- y; y2[i] <- y2[i] + 1
c(variacao_yhat_i = drop(H[i, ] %*% y2 - H[i, ] %*% y))
quantile(h, c(0.5, 0.9, 0.99))
c(acima_2p_n = sum(h > 2 * p / n), acima_3p_n = sum(h > 3 * p / n))
