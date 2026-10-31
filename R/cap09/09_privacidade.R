# Cap. 9 -- Privacidade diferencial: o mecanismo gaussiano
source(file.path("R", "comum", "gerador.R"))  # sorteios do Python
db <- read.csv(file.path("dados", "diabetes.csv"))
X <- as.matrix(db[, 1:10]); n <- nrow(X); p <- ncol(X)
X <- sweep(sweep(X, 2, apply(X, 2, min)), 2,
           apply(X, 2, function(v) diff(range(v))), "/")  # em [0, 1]
Delta <- sqrt(p)                       # sensibilidade da soma em l2

# sigma suficiente (proposição do texto):
# Phi((eta - eps) / sqrt(2 eta)) = delta
sigma_suf <- function(eps, delta) {
  z <- qnorm(delta); u <- (sqrt(2) * z + sqrt(2 * z^2 + 4 * eps)) / 2
  1 / (sqrt(2) * u)                    # em unidades de Delta
}
sigma_cla <- function(eps, delta) sqrt(2 * log(1.25 / delta)) / eps
sigma_exato <- function(eps, delta) {  # calibração de Balle e Wang
  h <- function(s) pnorm(1 / (2 * s) - eps * s) -
    exp(eps) * pnorm(-1 / (2 * s) - eps * s) - delta
  uniroot(h, c(1e-3, 1e3), tol = 1e-12)$root
}
delta <- 1e-5; eps <- c(0.1, 0.5, 1, 2, 4)
round(rbind(eps = eps, suficiente = sapply(eps, sigma_suf, delta),
            classico = sapply(eps, sigma_cla, delta),
            exato = sapply(eps, sigma_exato, delta)), 3)

# Duas publicações da soma: com todos e sem o paciente 1 (eps = 1)
sigma <- Delta * sigma_suf(1, delta); g <- gerador(19)
s_tudo <- colSums(X); s_sem <- colSums(X[-1, ])
r_tudo <- s_tudo + sigma * g$normal(p)
r_sem <- s_sem + sigma * g$normal(p)
c(sigma = sigma, erro_max_media = max(abs(r_tudo / n - s_tudo / n)),
  erro_exato = max(abs((s_tudo - s_sem) - X[1, ])),
  erro_com_ruido = max(abs((r_tudo - r_sem) - X[1, ])))
round(rbind(verdadeiro = X[1, ], reconstruido = r_tudo - r_sem), 2)
