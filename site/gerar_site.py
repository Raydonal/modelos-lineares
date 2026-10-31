"""Gera as páginas do site a partir de R/, python/, saidas/ e figuras/.

Cada exemplo mostra o código, a saída e as figuras que o acompanham no livro.
Nada é executado aqui: saídas e figuras vêm de `make exemplos` e `make figuras`,
os mesmos arquivos que alimentam o livro. Rode a partir de codigo/:
    python site/gerar_site.py && quarto render site
"""
import ast
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
    m = re.match(r'^(?:# |""")Cap\. ?\d+(?:: | -- )(.*?)\.?(?:""")?$', linha)
    if m is None:
        raise SystemExit(f"{caminho}: primeira linha sem título")
    return m.group(1)


def bloco(rotulo, caminho, texto=None):
    rel = caminho.relative_to(RAIZ)
    corpo = caminho.read_text(encoding="utf-8") if texto is None else texto
    return f'{CERCA}{{.{rotulo} filename="{rel}"}}\n{corpo.rstrip()}\n{CERCA}\n'


def aba(nome, rotulo, script, saida):
    rel = script.relative_to(RAIZ)
    return (f"## {nome}\n\n{bloco(rotulo, script)}\n"
            f"**Saída**\n\n{CERCA}{{.text}}\n"
            f'{saida.read_text(encoding="utf-8").rstrip()}\n{CERCA}\n\n'
            f"[Ver no GitHub]({REPO}/{rel})\n\n")


def funcoes_de_figura(cap):
    """Devolve {nome da figura: (script, código da função que a gera)}."""
    achadas = {}
    for f in sorted((RAIZ / "python" / cap).glob("fig_*.py")):
        fonte = f.read_text(encoding="utf-8")
        for no in ast.parse(fonte).body:
            if not isinstance(no, ast.FunctionDef):
                continue
            for c in ast.walk(no):
                if (isinstance(c, ast.Call)
                        and getattr(c.func, "id", "") == "salvar"):
                    achadas[c.args[2].value] = (
                        f, ast.get_source_segment(fonte, no))
    return achadas


def figura(cap, nome, legenda, funcoes):
    if nome not in funcoes:
        raise SystemExit(f"{cap}/{nome}: sem função que chame salvar()")
    script, codigo = funcoes[nome]
    destino = SITE / "imagens" / cap
    destino.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "130",
                    str(RAIZ / "figuras" / cap / f"{nome}.pdf"),
                    str(destino / nome)], check=True)
    rel = script.relative_to(RAIZ)
    rotulo = nome.replace("_", " ")
    return (f"#### Figura: {rotulo} {{#figura-{nome}}}\n\n"
            f"{bloco('python', script, codigo)}\n"
            f"Trecho de [`{rel}`]({REPO}/{rel}), que traz também as "
            "importações e o estilo.\n\n"
            f'![{legenda}](imagens/{cap}/{nome}.png){{fig-alt="Figura '
            f'{rotulo}"}}\n')


def carregar_figuras():
    por_exemplo = {}
    for linha in (SITE / "figuras.tsv").read_text(encoding="utf-8").splitlines():
        cap, nome, script, legenda = linha.split("\t")
        por_exemplo.setdefault((cap, script), []).append((nome, legenda))
    return por_exemplo


def exemplos(cap, por_exemplo, funcoes):
    partes = ["## Exemplos\n"]
    for r in sorted((RAIZ / "R" / cap).glob("[0-9]*.R")):
        py = RAIZ / "python" / cap / f"{r.stem}.py"
        said_r = RAIZ / "saidas" / cap / f"{r.stem}.R.txt"
        said_py = RAIZ / "saidas" / cap / f"{r.stem}.py.txt"
        partes.append(f"### {titulo_do_script(r)} {{#ex-{r.stem}}}\n\n"
                      '::: {.panel-tabset group="linguagem"}\n\n'
                      + aba("R", "r", r, said_r)
                      + aba("Python", "python", py, said_py) + ":::\n")
        for nome, legenda in por_exemplo.get((cap, r.stem), []):
            partes.append(figura(cap, nome, legenda, funcoes))
    soltas = por_exemplo.get((cap, ""), [])
    if soltas:
        partes.append("## Outras figuras\n")
        partes += [figura(cap, n, l, funcoes) for n, l in soltas]
    return "\n".join(partes)


def principal():
    (SITE / "imagens").exists() and shutil.rmtree(SITE / "imagens")
    por_exemplo = carregar_figuras()
    linhas = (SITE / "capitulos.tsv").read_text(encoding="utf-8")
    configuracao = (SITE / "_quarto.yml").read_text(encoding="utf-8")
    pendentes = {(c, s) for c, s in por_exemplo}
    for linha in linhas.strip().splitlines():
        cap, titulo = linha.split("\t")
        if f"{cap}.qmd" not in configuracao:
            raise SystemExit(f"Acrescente {cap}.qmd a site/_quarto.yml")
        n = int(cap[3:])
        cabecalho = (f'---\ntitle: "{n}. {titulo}"\n---\n\n'
                     f"Material reprodutível do Capítulo {n} do livro.\n\n")
        funcoes = funcoes_de_figura(cap)
        (SITE / f"{cap}.qmd").write_text(
            cabecalho + exemplos(cap, por_exemplo, funcoes), encoding="utf-8")
        print(">>", cap)
    feitas = {(c, r.stem) for c in {p[0] for p in pendentes}
              for r in (RAIZ / "R" / c).glob("[0-9]*.R")}
    orfas = {p for p in pendentes if p[1] and p not in feitas}
    if orfas:
        raise SystemExit(f"Figuras ligadas a scripts inexistentes: {orfas}")


if __name__ == "__main__":
    principal()
