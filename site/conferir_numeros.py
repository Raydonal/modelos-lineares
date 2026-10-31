"""Lista números acrescentados ao livro que não aparecem nas saídas.

Compara o texto de um capítulo com uma revisão do git de livro/. Uso:
    python site/conferir_numeros.py cap13 HEAD
"""
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LIVRO = RAIZ.parent / "livro"
NUM = re.compile(r"\d+(?:(?:\{,\}|[.,])\d+)?(?:[eE][-+]?\d+)?")


def normal(texto):
    return {n.replace("{,}", ".").replace(",", ".") for n in NUM.findall(texto)}


def principal(cap, rev):
    conhecido = set()
    for f in (RAIZ / "saidas" / cap).glob("*.txt"):
        conhecido |= normal(f.read_text(encoding="utf-8"))
    alterados = subprocess.run(
        ["git", "diff", "--unified=0", rev, "--", f"capitulos/{cap}"],
        cwd=LIVRO, capture_output=True, text=True, check=True).stdout
    novas = [l[1:] for l in alterados.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    velhas = [l[1:] for l in alterados.splitlines()
              if l.startswith("-") and not l.startswith("---")]
    conhecido |= normal("\n".join(velhas))
    # o texto pode arredondar: 832.13 vale se a saída traz 832.1341
    suspeitos = sorted(n for n in normal("\n".join(novas)) - conhecido
                       if not any(c.startswith(n) for c in conhecido))
    # inteiros pequenos são quase sempre rótulos, contagens ou expoentes
    suspeitos = [n for n in suspeitos if "." in n or len(n) > 2]
    print(cap, "números sem origem:", suspeitos or "nenhum")


if __name__ == "__main__":
    principal(sys.argv[1], sys.argv[2])
