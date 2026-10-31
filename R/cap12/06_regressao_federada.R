# Cap. 12 -- Regressão federada: combinar regiões sem juntar os dados
w <- read.csv(file.path("dados", "wage1.csv"))
w$regiao <- with(w, ifelse(northcen == 1, "centro-norte",
              ifelse(south == 1, "sul", ifelse(west == 1, "oeste",
                                                "nordeste"))))
X <- with(w, cbind(1, educ, exper, exper^2 / 100, tenure, female))
y <- log(w$wage); p <- ncol(X); n <- nrow(X)
# Cada sítio envia só X_k'X_k, X_k'y_k, y_k'y_k e n_k
sitios <- split(seq_len(n), w$regiao)
env <- lapply(sitios, function(i) list(XtX = crossprod(X[i, ]),
  Xty = crossprod(X[i, ], y[i]), yty = sum(y[i]^2), n = length(i)))
sapply(env, `[[`, "n")
S <- Reduce(`+`, lapply(env, `[[`, "XtX"))
b_fed <- solve(S, Reduce(`+`, lapply(env, `[[`, "Xty")))
b_mq <- solve(crossprod(X), crossprod(X, y))
max(abs(b_fed - b_mq))                        # idênticos
sqr <- sum(sapply(env, `[[`, "yty")) - sum(b_fed * crossprod(X, y))
s2 <- sqr / (n - p)
# Alternativa: média simples das estimativas de cada região
bk <- sapply(env, function(e) solve(e$XtX, e$Xty))
b_med <- rowMeans(bk)
C_fed <- s2 * solve(S)
C_med <- s2 * Reduce(`+`, lapply(env, function(e) solve(e$XtX))) / 16
j <- 2                                        # educação
round(c(educ_fed = b_fed[j], ep_fed = sqrt(C_fed[j, j]),
        educ_med = b_med[j], ep_med = sqrt(C_med[j, j]),
        razao_var = C_med[j, j] / C_fed[j, j]), 4)
round(bk[j, ], 4)                             # educação por região
signif(eigen(C_med - C_fed)$values / s2, 3)   # Loewner: todos >= 0
# H: mesmo beta nas 4 regiões? (teste de Chow, Cap. 10)
sqr_k <- sum(sapply(names(env), function(k) { e <- env[[k]]
  e$yty - sum(bk[, k] * e$Xty) }))
F <- ((sqr - sqr_k) / (3 * p)) / (sqr_k / (n - 4 * p))
round(c(F = F, gl1 = 3 * p, gl2 = n - 4 * p,
        p_valor = pf(F, 3 * p, n - 4 * p, lower.tail = FALSE)), 3)
