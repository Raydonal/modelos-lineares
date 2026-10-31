# Cap. 13: MQ2E como MQG e o teste de Hausman (dados de Mroz)
m <- read.csv(file.path("dados", "mroz.csv"))
y <- m$lwage; n <- length(y); X <- cbind(1, educ = m$educ)
Zs <- list(pai = cbind(1, m$fatheduc),
           pai_mae = cbind(1, m$fatheduc, m$motheduc))
bmq <- drop(solve(crossprod(X), crossprod(X, y)))
s2 <- sum((y - X %*% bmq)^2) / (n - 2)        # MQ é eficiente sob H0
t(sapply(Zs, function(Z) {
  PZ <- Z %*% solve(crossprod(Z), t(Z))
  biv <- drop(solve(t(X) %*% PZ %*% X, t(X) %*% PZ %*% y))
  # MQG no modelo Z'y = Z'X b + Z'e, com Cov(Z'e) = s2 Z'Z
  W <- solve(crossprod(Z))
  bg <- solve(t(X) %*% Z %*% W %*% t(Z) %*% X,
              t(X) %*% Z %*% W %*% t(Z) %*% y)
  vd <- s2 * (solve(t(X) %*% PZ %*% X) - solve(crossprod(X)))[2, 2]
  H <- (biv[[2]] - bmq[[2]])^2 / vd
  round(c(dif_mqg = max(abs(bg - biv)), mq = bmq[[2]],
    mq2e = biv[[2]], ep_dif = sqrt(vd), H = H,
    p = pchisq(H, 1, lower.tail = FALSE)), 4)
}))
