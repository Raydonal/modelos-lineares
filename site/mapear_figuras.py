"""Liga cada figura ao exemplo que ela acompanha no livro e grava a legenda.

O texto do livro não é versionado aqui: este script só roda onde `../livro`
existe e grava site/figuras.tsv, que o site usa. Rode de novo quando entrar
uma figura nova:  python site/mapear_figuras.py
"""
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LIVRO = RAIZ.parent / "livro"
TOKEN = re.compile(
    r"\\(?P<bloco>section|subsection|aplicacao)\b"
    r"|\\codigoR(?:\[[^\]]*\])?\{R/(?P<cap_s>cap\d\d)/(?P<script>[^}]+)\.R\}"
    r"|\\includegraphics(?:\[[^\]]*\])?\{(?P<cap_f>cap\d\d)/(?P<fig>[^}]+)\}")


def legenda_em_latex(texto, inicio):
    i = texto.find(r"\caption{", inicio)
    fim_fig = texto.find(r"\end{figure}", inicio)
    if i < 0 or (0 <= fim_fig < i):
        return ""
    i += len(r"\caption{")
    nivel, j = 1, i
    while nivel:
        nivel += {"{": 1, "}": -1}.get(texto[j], 0)
        j += 1
    return texto[i:j - 1]


def para_markdown(latex, macros):
    latex = re.sub(r"\\label\{[^}]*\}", "", latex)
    latex = re.sub(r"\\[Cc]ref\{[^}]*\}", "o texto do livro", latex)
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "markdown", "--wrap=none"],
                       input=macros + "\n" + latex, capture_output=True,
                       text=True, check=True)
    md = " ".join(r.stdout.split()).replace(r"\symbf", r"\boldsymbol")
    md = trocar_delimitador(md, r"\abs{", r"\lvert ", r"\rvert")
    md = trocar_delimitador(md, r"\norm{", r"\lVert ", r"\rVert")
    md = re.sub(r"\\tr\b", r"\\operatorname{tr}", md)
    return re.sub(r"\\E\b", r"\\mathrm{E}", md)


def trocar_delimitador(md, macro, abre, fecha):
    while (i := md.find(macro)) >= 0:
        nivel, j = 1, i + len(macro)
        while nivel:
            nivel += {"{": 1, "}": -1}.get(md[j], 0)
            j += 1
        md = md[:i] + abre + md[i + len(macro):j - 1] + fecha + md[j:]
    return md


def eventos(cap):
    principal = (LIVRO / "capitulos" / f"{cap}.tex").read_text(encoding="utf-8")
    for nome in re.findall(r"^\s*\\input\{([^}]+)\}", principal, re.M):
        arq = LIVRO / (nome if nome.endswith(".tex") else f"{nome}.tex")
        texto = arq.read_text(encoding="utf-8")
        texto = re.sub(r"(?<!\\)%.*", "", texto)
        for m in TOKEN.finditer(texto):
            yield m, texto


def principal():
    macros = "\n".join(l.split("%")[0] for l in
                       (LIVRO / "preambulo" / "notacao.tex").read_text(
                           encoding="utf-8").splitlines()
                       if l.lstrip().startswith(r"\newcommand"))
    linhas = []
    for tex in sorted((LIVRO / "capitulos").glob("cap[0-9][0-9].tex")):
        cap = tex.stem
        blocos, atual = [], []
        for m, texto in eventos(cap):
            if m["bloco"]:
                blocos.append(atual); atual = []
            elif m["script"]:
                atual.append(("s", m["cap_s"], m["script"]))
            else:
                atual.append(("f", m["cap_f"], m["fig"],
                              legenda_em_latex(texto, m.end())))
        blocos.append(atual)
        for bloco in blocos:
            for i, ev in enumerate(bloco):
                if ev[0] != "f":
                    continue
                antes = [e[2] for e in reversed(bloco[:i])
                         if e[0] == "s" and e[1] == ev[1]]
                depois = [e[2] for e in bloco[i:]
                          if e[0] == "s" and e[1] == ev[1]]
                alvo = (antes or depois or [""])[0]
                linhas.append((ev[1], ev[2], alvo, para_markdown(ev[3], macros)))
    vistos, saida = set(), []
    for linha in linhas:
        if linha[:2] not in vistos:
            vistos.add(linha[:2]); saida.append("\t".join(linha))
    (RAIZ / "site" / "figuras.tsv").write_text("\n".join(saida) + "\n",
                                               encoding="utf-8")
    print(len(saida), "figuras;", sum(not l[2] for l in linhas), "sem exemplo")


if __name__ == "__main__":
    principal()
