# Cap. 5: Suavizadores lineares: projeções e não projeções (dados mcycle)
mc <- read.csv(file.path("dados", "mcycle.csv")); x <- mc$times; y <- mc$accel; n <- length(y)
tr <- function(A) sum(diag(A))
base <- function(K) {                         # B-splines cúbicas, K nós interiores equiespaçados
  h <- (max(x) - min(x)) / (K + 1)
  splines::splineDesign(min(x) + h * (-3:(K + 4)), x, ord = 4)
}
resumo <- function(S) {
  r <- y - S %*% y
  c(gl = tr(S), tr_StS = tr(crossprod(S)), erro_idemp = max(abs(S %*% S - S)),
    assimetria = max(abs(S - t(S))), max_Sii = max(diag(S)),
    CV = mean((r / (1 - diag(S)))^2), GCV = mean(r^2) / (1 - tr(S) / n)^2)
}
# (1) Spline de regressão: projeção ortogonal sobre C(B); nós escolhidos por validação cruzada
proj <- function(B) B %*% solve(crossprod(B), t(B))
cv_K <- sapply(2:20, function(K) resumo(proj(base(K)))["CV"])
K_opt <- (2:20)[which.min(cv_K)]
round(rbind(K = 2:20, CV = cv_K)[, c(1, 4, 8, 12, 16, 19)], 1)
S_reg <- proj(base(K_opt))

# (2) P-spline: 20 nós e penalidade de segundas diferenças, lambda por GCV
B <- base(20); Dd <- diff(diag(ncol(B)), differences = 2)
S_pen <- function(l) B %*% solve(crossprod(B) + l * crossprod(Dd), t(B))
lams <- 10^seq(-2, 4, length.out = 121)
l_opt <- lams[which.min(sapply(lams, function(l) resumo(S_pen(l))["GCV"]))]
S_p <- S_pen(l_opt)
ev <- eigen(S_p, symmetric = TRUE)$values

# (3) Núcleo gaussiano de Nadaraya-Watson, janela por validação cruzada
nw <- function(b) { W <- exp(-0.5 * outer(x, x, "-")^2 / b^2); W / rowSums(W) }
bs <- seq(0.5, 4, by = 0.05)
b_opt <- bs[which.min(sapply(bs, function(b) resumo(nw(b))["CV"]))]
S_nw <- nw(b_opt)
c(K_opt = K_opt, lambda_opt = l_opt, janela_opt = b_opt)
round(rbind(spline_regressao = resumo(S_reg), p_spline = resumo(S_p), nadaraya_watson = resumo(S_nw)), 4)
round(range(ev), 6)                                     # autovalores do P-spline em [0, 1]
c(soma_linhas_nw = max(abs(rowSums(S_nw) - 1)), soma_linhas_p = max(abs(rowSums(S_p) - 1)))

# Validação cruzada sem reajuste: confere numa observação
i <- 50; W <- exp(-0.5 * (x[i] - x[-i])^2 / b_opt^2)
c(direto = sum(W * y[-i]) / sum(W), atalho = drop(S_nw[i, ] %*% y - S_nw[i, i] * y[i]) / (1 - S_nw[i, i]))
