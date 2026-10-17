"""CLI: classifica um documento fiscal a partir de um arquivo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classifier import ClassificadorDocumentosFiscais, dataset_exemplo
from .pipeline import classifica_arquivo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classifica documentos fiscais (NF-e / boleto)."
    )
    parser.add_argument("arquivo", help="Caminho do PDF ou arquivo de texto.")
    args = parser.parse_args(argv)

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"Erro: arquivo não encontrado: {caminho}", file=sys.stderr)
        return 1

    classificador = ClassificadorDocumentosFiscais()
    textos, tipos = dataset_exemplo()
    classificador.treina(textos, tipos)

    classificacao = classifica_arquivo(caminho, classificador)
    print(f"Arquivo: {caminho}")
    print(f"Tipo previsto: {classificacao.tipo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
