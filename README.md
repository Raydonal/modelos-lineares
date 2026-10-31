# Modelos Lineares: Teoria, Geometria e Aplicações: código companheiro

Código em **R** e **Python** que reproduz todos os exemplos, saídas e figuras do livro
*Modelos Lineares: Teoria, Geometria e Aplicações* (título provisório), de
Raydonal Ospina, Cristiano Ferraz e André Leite.

**Site:** <https://raydonal.github.io/modelos-lineares/> (código, saídas e figuras de cada exemplo).

## Estrutura

```
R/capXX/        exemplos em R          (NN_nome.R)   e figuras (fig_*.R)
python/capXX/   exemplos em Python     (NN_nome.py)  e figuras (fig_*.py)
R/comum/, python/comum/   estilo gráfico e utilitários compartilhados
dados/          conjuntos de dados em CSV (gerados por `make dados`)
figuras/capXX/  figuras em PDF usadas no livro
saidas/capXX/   saídas dos exemplos, incluídas no livro tal como geradas
site/           site Quarto: mostra exemplos, saídas e figuras (`make site`)
```

Cada exemplo existe nas duas linguagens com o mesmo nome. Os scripts devem ser executados
**a partir desta pasta** (raiz do repositório).

## Ambiente

**Python** (≥ 3.11):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**R** (≥ 4.3), com as versões fixadas em `renv.lock`:

```r
install.packages("renv")
renv::restore()
```

**Docker** (ambiente completo, R + Python):

```bash
docker build -t modelos-lineares .
docker run --rm -v "$PWD":/livro modelos-lineares make tudo
```

## Uso

```bash
make dados      # exporta os dados para dados/
make figuras    # gera todas as figuras
make exemplos   # roda todos os exemplos e grava as saídas em saidas/
make tudo       # tudo acima
make cap01      # só o capítulo 1 (figuras + exemplos)
```

Use `make PYTHON=/caminho/para/python` para escolher o interpretador.

## Dados

| Arquivo | Origem | Capítulo |
|---|---|---|
| `yates_oats.csv` | `agridat::yates.oats` (Yates, 1935) | 1 |
| `wage1.csv` | `wooldridge::wage1` (Wooldridge) | 1 |
| `digitos.csv` | `sklearn.datasets.load_digits` | 1 |
| `ukgas.csv` | `datasets::UKgas` | 1 |
| `longley.csv` | `datasets::longley` (Longley, 1967; NIST StRD) | 2 |
| `diabetes.csv` | `sklearn.datasets.load_diabetes` (Efron et al., 2004) | 3 |
| `nfl.csv` | `LearnBayes::footballscores` (NFL, 1981, 1983 e 1984) | 4 |
| `seatbelts.csv` | `datasets::Seatbelts` (Harvey e Durbin, 1986) | 5 |
| `mroz.csv` | `wooldridge::mroz`, mulheres na força de trabalho (Mroz, 1987) | 5 |
| `mcycle.csv` | `MASS::mcycle` (Silverman, 1985) | 5 |
| `yan_trigo.csv` | `agridat::yan.winterwheat` (Yan e Kang, 2002) | 6 |
| `austen.csv` | `janeaustenr::austen_books()`, os seis romances de Jane Austen | 1, 5, 6, 7 |
| `eustock.csv` | `datasets::EuStockMarkets`, índices DAX, SMI, CAC e FTSE (1991–1998) | 9 |
| `airquality.csv` | `datasets::airquality`, qualidade do ar em Nova York (1973) | 9 |

## Site

`make site` gera as páginas a partir dos scripts, das saídas e das figuras já gravados e renderiza o site
em `site/_site/` (requer [Quarto](https://quarto.org) e `pdftocairo`, do poppler). Nada é executado de novo:
depois de mudar um script, rode `make capXX` antes. As páginas `site/capXX.qmd` são geradas e não devem ser
editadas. Para publicar: `make publicar` envia `site/_site` à branch `gh-pages`, servida pelo GitHub Pages.

## Licença

Código sob licença MIT (ver `LICENSE`). Os dados pertencem às suas fontes originais.
