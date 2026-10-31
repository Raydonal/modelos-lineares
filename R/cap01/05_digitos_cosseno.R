# Cap. 1 -- Similaridade cosseno e classificação pelo centróide mais próximo
dig <- read.csv(file.path("dados", "digitos.csv"))
X <- as.matrix(dig[, -1])                      # 1797 imagens x 64 pixels
rotulo <- dig$digito

treino <- 1:1200                               # partição fixa (igual à do Python)
teste  <- 1201:nrow(X)

normalizar <- function(M) M / sqrt(rowSums(M^2))   # cada linha com norma 1

# centróides (médias por classe) no conjunto de treino
centr <- rowsum(X[treino, ], rotulo[treino]) / as.vector(table(rotulo[treino]))
S <- normalizar(centr) %*% t(normalizar(centr))    # matriz de cossenos 10 x 10
round(S[1:4, 1:4], 3)

# classificação: rótulo do centróide de maior cosseno
cos_teste <- normalizar(X[teste, ]) %*% t(normalizar(centr))
pred <- as.integer(colnames(cos_teste)[max.col(cos_teste)])
mean(pred == rotulo[teste])                         # acurácia no teste

# matriz de confusão (linhas: verdadeiro; colunas: previsto)
table(verdadeiro = rotulo[teste], previsto = pred)
