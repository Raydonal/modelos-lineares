# Cap. 1: Correlação amostral como cosseno entre vetores centrados
dados <- read.csv(file.path("dados", "wage1.csv"))
y <- log(dados$wage)                   # log do salário-hora
x <- dados$educ                        # anos de escolaridade

cosseno <- function(u, v) sum(u * v) / sqrt(sum(u^2) * sum(v^2))
centrar <- function(v) v - mean(v)

c(n          = length(y),
  cos_bruto  = cosseno(x, y),                    # vetores não centrados
  cos_centr  = cosseno(centrar(x), centrar(y)),  # = correlação
  cor        = cor(x, y),
  angulo_graus = acos(cor(x, y)) * 180 / pi)
