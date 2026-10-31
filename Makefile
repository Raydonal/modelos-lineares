# =====================================================================
#  Makefile do código companheiro
#    make dados        prepara os conjuntos de dados em dados/
#    make figuras      gera todas as figuras do livro em figuras/
#    make exemplos     roda os exemplos e grava as saídas em saidas/capXX/
#                      (o livro inclui essas saídas diretamente)
#    make tudo         dados + figuras + exemplos
#    make cap01        apenas o capítulo 1 (idem cap02, ...)
#    make site         gera o site em site/_site/ (requer quarto)
#    make publicar     gera o site e envia à branch gh-pages
#
#  Variáveis: PYTHON (padrão: python3) e RSCRIPT (padrão: Rscript)
# =====================================================================
PYTHON  ?= python3
RSCRIPT ?= Rscript
CAPS    := $(patsubst R/%,%,$(wildcard R/cap*))

.PHONY: tudo dados figuras exemplos site publicar limpar $(CAPS)

tudo: dados figuras exemplos

dados:
	$(RSCRIPT) R/preparar_dados.R
	$(PYTHON) python/preparar_dados.py

figuras:
	@for f in python/cap*/fig_*.py; do echo ">> $$f"; $(PYTHON) $$f || exit 1; done
	@for f in $$(ls R/cap*/fig_*.R 2>/dev/null); do echo ">> $$f"; $(RSCRIPT) $$f || exit 1; done

exemplos:
	@for f in $$(ls R/cap*/[0-9]*.R); do c=$$(basename $$(dirname $$f)); mkdir -p saidas/$$c; \
	  echo ">> $$f"; $(RSCRIPT) $$f > saidas/$$c/$$(basename $$f .R).R.txt 2>&1 || exit 1; done
	@for f in $$(ls python/cap*/[0-9]*.py); do c=$$(basename $$(dirname $$f)); mkdir -p saidas/$$c; \
	  echo ">> $$f"; $(PYTHON) $$f > saidas/$$c/$$(basename $$f .py).py.txt 2>&1 || exit 1; done

$(CAPS):
	@mkdir -p saidas/$@
	@for f in $$(ls python/$@/fig_*.py 2>/dev/null); do echo ">> $$f"; $(PYTHON) $$f || exit 1; done
	@for f in $$(ls R/$@/fig_*.R 2>/dev/null); do echo ">> $$f"; $(RSCRIPT) $$f || exit 1; done
	@for f in $$(ls R/$@/[0-9]*.R); do echo ">> $$f"; \
	  $(RSCRIPT) $$f > saidas/$@/$$(basename $$f .R).R.txt 2>&1 || exit 1; done
	@for f in $$(ls python/$@/[0-9]*.py); do echo ">> $$f"; \
	  $(PYTHON) $$f > saidas/$@/$$(basename $$f .py).py.txt 2>&1 || exit 1; done

site:
	$(PYTHON) site/gerar_site.py
	quarto render site

publicar: site
	bash site/publicar.sh

limpar:
	rm -rf figuras/cap* saidas/cap*
