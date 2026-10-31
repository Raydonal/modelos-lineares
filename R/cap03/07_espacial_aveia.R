# Cap. 3: Verossimilhança gaussiana com correlação espacial: log-determinantes e sistemas
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
y <- aveia$yield
X <- model.matrix(~ factor(block) + factor(nitro) + factor(gen), aveia)   # 72 x 11
link <- 0.2012                                        # metros por link
coord <- cbind(x = (aveia$col - 1) * 44 * link,      # parcelas de 28,4 x 44 links
               y = (aveia$row - 1) * 28.4 * link)
D <- as.matrix(dist(coord))                           # distâncias entre parcelas (m)
n <- length(y)

# Log-verossimilhança perfilada em phi: Sigma = sigma2 * R(phi), R_ij = exp(-d_ij / phi)
logvero <- function(phi) {
  R <- if (phi == 0) diag(n) else exp(-D / phi)
  RiX <- solve(R, X); Riy <- solve(R, y)              # sistemas, nunca inversas
  beta <- solve(crossprod(X, RiX), crossprod(X, Riy)) # estimador de MQ generalizados
  r <- y - X %*% beta
  sigma2 <- sum(r * solve(R, r)) / n
  ld <- determinant(R, logarithm = TRUE)$modulus      # log det R via LU
  as.numeric(-n / 2 * log(2 * pi * sigma2) - ld / 2 - n / 2)
}
phis <- c(0, 1, 2, 4, 8, 15, 30, 45, 60)
ll <- sapply(phis, logvero)
round(cbind(phi = phis, logvero = ll), 2)
opt <- optimize(logvero, c(0.5, 60), maximum = TRUE)
c(phi_hat = opt$maximum, ganho = opt$objective - logvero(0))   # 2 x ganho ~ razão de verossimilhanças
