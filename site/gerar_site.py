"""Gera as páginas do site a partir de R/, python/, saidas/ e figuras/.

Nada aqui executa os exemplos: o site mostra as saídas já gravadas por
`make exemplos`, as mesmas que alimentam o livro. Rode a partir de codigo/:
    python site/gerar_site.py && quarto render site
"""
import re
import shutil
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SITE = RAIZ / "site"
REPO = "https://github.com/Raydonal/modelos-lineares/blob/main"
CERCA = "`````"          # mais crases do que qualquer script usa


def titulo_do_script(caminho):
    linha = caminho.read_text(encoding="utf-8").splitlines()[0]
    m = re.match(r'^(?:# |""")Cap\. ?\d+ -- (.*?)\.?(?:""")?$', linha)
    if m is None:
        raise SystemExit(f"{caminho}: primeira linha sem título")
    return m.group(1)


def bloco(rotulo, caminho):
    rel = caminho.relative_to(RAIZ)
    return (f'{CERCA}{{.{rotulo} filename="{rel}"}}\n'
            f'{caminho.read_text(encoding="utf-8").rstrip()}\n{CERCA}\n')


def aba(nome, rotulo, script, saida):
    rel = script.relative_to(RAIZ)
    return (f"## {nome}\n\n{bloco(rotulo, script)}\n"
            f"**Saída**\n\n{CERCA}{{.text}}\n"
            f'{saida.read_text(encoding="utf-8").rstrip()}\n{CERCA}\n\n'
            f"[Ver no GitHub]({REPO}/{rel})\n\n")


def exemplos(cap):
    partes = ["## Exemplos\n"]
    for r in sorted((RAIZ / "R" / cap).glob("[0-9]*.R")):
        py = RAIZ / "python" / cap / f"{r.stem}.py"
        said_r = RAIZ / "saidas" / cap / f"{r.stem}.R.txt"
        said_py = RAIZ / "saidas" / cap / f"{r.stem}.py.txt"
        partes.append(f"### {titulo_do_script(r)} {{#ex-{r.stem}}}\n\n"
                      '::: {.panel-tabset group="linguagem"}\n\n'
                      + aba("R", "r", r, said_r)
                      + aba("Python", "python", py, said_py) + ":::\n")
    return "\n".join(partes)


def figuras(cap):
    pdfs = sorted((RAIZ / "figuras" / cap).glob("*.pdf"))
    if not pdfs:
        return ""
    destino = SITE / "imagens" / cap
    destino.mkdir(parents=True, exist_ok=True)
    partes = ["## Figuras\n"]
    for pdf in pdfs:
        png = destino / pdf.stem
        subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "130",
                        str(pdf), str(png)], check=True)
        nome = pdf.stem.replace("_", " ")
        partes.append(f"### {nome} {{#figura-{pdf.stem}}}\n\n"
                      f'![](imagens/{cap}/{pdf.stem}.png){{fig-alt="Figura '
                      f'{nome} do capítulo"}}\n')
    partes.append("### Código das figuras\n")
    for f in sorted((RAIZ / "python" / cap).glob("fig_*.py")):
        partes.append(f"::: {{.callout-note collapse=true title=\"{f.name}\"}}"
                      f"\n{bloco('python', f)}:::\n")
    return "\n".join(partes)


def principal():
    (SITE / "imagens").exists() and shutil.rmtree(SITE / "imagens")
    linhas = (SITE / "capitulos.tsv").read_text(encoding="utf-8")
    configuracao = (SITE / "_quarto.yml").read_text(encoding="utf-8")
    for linha in linhas.strip().splitlines():
        cap, titulo = linha.split("\t")
        if f"{cap}.qmd" not in configuracao:
            raise SystemExit(f"Acrescente {cap}.qmd a site/_quarto.yml")
        n = int(cap[3:])
        cabecalho = (f'---\ntitle: "{n}. {titulo}"\n---\n\n'
                     f"Material reprodutível do Capítulo {n} do livro.\n\n")
        (SITE / f"{cap}.qmd").write_text(
            cabecalho + exemplos(cap) + "\n" + figuras(cap), encoding="utf-8")
        print(">>", cap)


if __name__ == "__main__":
    principal()
