# Cap. 14: Falta de ajuste e o teste de aditividade de Tukey
pr <- function(...) cat(sprintf(...), "\n", sep = "")
x <- c(0, 0, 1, 1, 2, 2); y <- c(1, 3, 2, 4, 7, 7)
X <- cbind(1, x); b <- solve(crossprod(X), crossprod(X, y))
ybar <- ave(y, x); yh <- drop(X %*% b)
sqp <- sum((y - ybar)^2); sqf <- sum((ybar - yh)^2)
Fl <- (sqf / 1) / (sqp / 3)
pr("b %.2f %.2f SQ falta %.2f SQ puro %.2f F %.2f p %.3f", b[1],
   b[2], sqf, sqp, Fl, pf(Fl, 1, 3, lower.tail = FALSE))
# Tukey (1949): um grau de liberdade para não aditividade
Y <- rbind(c(2, 1, 3), c(1, 5, 6), c(3, 6, 9))
tukey <- function(Y) {
  a <- rowMeans(Y) - mean(Y); bb <- colMeans(Y) - mean(Y)
  E <- Y - outer(a, bb, "+") - mean(Y)
  D <- sum(outer(a, bb) * Y)^2 / (sum(a^2) * sum(bb^2))
  gl <- (nrow(Y) - 1) * (ncol(Y) - 1) - 1
  Fv <- D / ((sum(E^2) - D) / gl)
  list(a = a, b = bb, sqr = sum(E^2), D = D, F = Fv, gl = gl,
       p = pf(Fv, 1, gl, lower.tail = FALSE))
}
t <- tukey(Y)
pr("a %.3f %.3f %.3f b %.3f %.3f %.3f", t$a[1], t$a[2], t$a[3],
   t$b[1], t$b[2], t$b[3])
pr("SQRes %.4f D %.4f F %.2f gl %d p %.3f", t$sqr, t$D, t$F, t$gl,
   t$p)
