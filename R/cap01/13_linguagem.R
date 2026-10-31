# Cap. 1 -- Modelos de linguagem pela álgebra linear: tokens (BPE), embeddings e busca por similaridade
d <- read.csv(file.path("dados", "austen.csv"))
palavras <- function(txt) unlist(regmatches(tolower(txt), gregexpr("[a-z]+", tolower(txt))))

# (1) Tokenização por pares de bytes (BPE), treinada em Orgulho e Preconceito
pp <- palavras(d$texto[d$livro == "Pride & Prejudice"])
freq <- table(pp); freq <- freq[freq >= 2]                  # tipos que aparecem 2+ vezes
f <- as.numeric(freq)
# cada palavra vira uma cadeia de símbolos separados por espaços; "_" marca o fim da palavra
w <- paste0(" ", gsub("(.)", "\\1 ", names(freq)), "_ ")
fundir <- function(w, a, b)                                 # substitui o par (a, b) pelo símbolo ab
  gsub(paste0("(?<= )", a, " ", b, "(?= )"), paste0(a, b), w, perl = TRUE)
regras <- character(0)
for (k in 1:200) {
  s <- strsplit(trimws(w), " ")
  n <- lengths(s)
  a <- unlist(lapply(s, function(x) x[-length(x)])); b <- unlist(lapply(s, function(x) x[-1]))
  cont <- rowsum(rep(f, n - 1), paste(a, b))                # frequência de cada par adjacente
  ordem <- order(-cont[, 1], rownames(cont), method = "radix")   # empates: ordem lexicográfica
  par <- strsplit(rownames(cont)[ordem[1]], " ")[[1]]
  regras <- c(regras, paste(par, collapse = " "))
  w <- fundir(w, par[1], par[2])
}
head(regras, 12)
tail(regras, 6)
vocab <- sort(unique(c(unlist(strsplit(trimws(w), " ")), letters, "_")), method = "radix")
length(vocab)
tokenizar <- function(p) {
  x <- paste0(" ", gsub("(.)", "\\1 ", p), "_ ")
  for (r in strsplit(regras, " ")) x <- fundir(x, r[1], r[2])
  strsplit(trimws(x), " ")[[1]]
}
frase <- "the unbelievably proud gentleman danced reluctantly"
toks <- unlist(lapply(palavras(frase), tokenizar))
toks
c(palavras = length(palavras(frase)), tokens = length(toks), caracteres = nchar(frase))

# (2) Embeddings: a consulta a uma tabela é um produto pela matriz one-hot
source(file.path("R", "comum", "gerador.R")); g <- gerador(1)
V <- length(vocab); dm <- 8
E <- matriz_normal(g, V, dm)                                # matriz de embeddings (V x d)
ids <- match(toks, vocab)                                   # índices dos tokens
O <- matrix(0, length(ids), V); O[cbind(seq_along(ids), ids)] <- 1
c(V = V, d = dm, erro_OE_vs_indices = max(abs(O %*% E - E[ids, ])))

# (3) Busca por similaridade (a etapa de recuperação de um RAG): capítulos como vetores TF-IDF
cap <- ave(as.numeric(grepl("^chapter [0-9ivxlc]+", d$texto, ignore.case = TRUE)), d$livro, FUN = cumsum)
doc <- paste(d$livro, "- cap.", cap)[cap > 0]
txt <- tapply(d$texto[cap > 0], factor(doc, levels = unique(doc)), paste, collapse = " ")
tok <- lapply(txt, palavras)
termos <- sort(unique(unlist(tok)), method = "radix")
tf <- t(sapply(tok, function(w) tabulate(match(w, termos), length(termos))))
idf <- log(nrow(tf) / colSums(tf > 0))
X <- sweep(tf, 2, idf, "*")
X <- X / sqrt(rowSums(X^2))                                 # linhas de norma 1
dim(X)
consulta <- "a ball at netherfield where elizabeth dances with darcy"
q <- tabulate(match(palavras(consulta), termos), length(termos)) * idf
q <- q / sqrt(sum(q^2))
cosseno <- drop(X %*% q)
round(head(sort(cosseno, decreasing = TRUE), 4), 3)
