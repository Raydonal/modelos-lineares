# Cap. 8 -- Máxima verossimilhança da normal multivariada pelo cálculo matricial (dados wage1)
w <- read.csv(file.path("dados", "wage1.csv"))
X <- cbind(lsal = log(w$wage), educ = w$educ, exper = w$exper, perm = w$tenure); n <- nrow(X); p <- ncol(X)
loglik <- function(mu, Sigma) {
  R <- chol(Sigma); Z <- backsolve(R, t(X) - mu, transpose = TRUE)   # Z = R^{-T} (x_i - mu)
  -n * p / 2 * log(2 * pi) - n * sum(log(diag(R))) - sum(Z^2) / 2
}
mu_hat <- colMeans(X); Ac <- crossprod(sweep(X, 2, mu_hat)); Sigma_hat <- Ac / n
round(Sigma_hat, 3)
# Gradiente em relação a Sigma: G = (1/2) Sigma^{-1} (A - n Sigma) Sigma^{-1} (nulo no máximo)
grad <- function(S) { Si <- solve(S); 0.5 * Si %*% (Ac - n * S) %*% Si }
c(norma_grad_no_mle = norm(grad(Sigma_hat), "F"), norma_grad_em_cov = norm(grad(cov(X)), "F"))
# Conferência do gradiente por diferenças finitas (direção simétrica sorteada)
source(file.path("R", "comum", "gerador.R")); g <- gerador(84)
E <- matriz_normal(g, p, p); E <- (E + t(E)) / 2; h <- 1e-6; S0 <- cov(X) * 1.3
c(numerica = (loglik(mu_hat, S0 + h * E) - loglik(mu_hat, S0 - h * E)) / (2 * h), formula = sum(grad(S0) * E))
# Perturbações de Sigma_hat (em direções simétricas) sempre reduzem a verossimilhança
l0 <- loglik(mu_hat, Sigma_hat)
dif <- replicate(1000, { E <- matriz_normal(g, p, p); E <- 0.02 * (E + t(E)) / 2 * sqrt(mean(diag(Sigma_hat)))
  S <- Sigma_hat + E; if (min(eigen(S, symmetric = TRUE)$values) <= 0) NA else loglik(mu_hat, S) - l0 })
c(max_diferenca = max(dif, na.rm = TRUE), loglik_max = l0)
# Correlações implicadas pelo estimador
round(cov2cor(Sigma_hat), 3)
