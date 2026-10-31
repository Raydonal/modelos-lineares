# Cap. 1: Decomposição ortogonal da produção de aveia (Yates, 1935)
aveia <- read.csv(file.path("dados", "yates_oats.csv"))
y <- aveia$yield
n <- length(y)
dose <- factor(aveia$nitro)                    # 4 doses de nitrogênio

# Base ortogonal: indicadores das doses (colunas mutuamente ortogonais)
D <- model.matrix(~ dose - 1)
crossprod(D)                                   # diagonal: <d_i, d_j> = 0, i != j

# Projeção sobre span{1} e sobre span{d_1, ..., d_4}
y_med <- rep(mean(y), n)                                   # média geral
y_trt <- D %*% (crossprod(D, y) / colSums(D))              # médias por dose

sq <- c(total      = sum((y - y_med)^2),
        tratamento = sum((y_trt - y_med)^2),
        residuo    = sum((y - y_trt)^2))
sq
sq["tratamento"] + sq["residuo"] - sq["total"]             # Pitágoras: ~ 0
sum((y_trt - y_med) * (y - y_trt))                         # ortogonalidade: ~ 0

options(show.signif.stars = FALSE)
anova(lm(y ~ dose))                            # mesmas somas de quadrados
