"""Extração de texto de documentos fiscais (PDF via pdfminer.six)."""

from __future__ import annotations

from pathlib import Path


def extrai_texto(caminho: str | Path) -> str:
    """Extrai o texto de um arquivo.

    Arquivos ``.pdf`` são lidos com pdfminer.six; demais arquivos são lidos
    como texto simples, o que mantém os testes determinísticos sem depender
    de um PDF real.
    """
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    sufixo = caminho_arquivo.suffix.lower()
    if sufixo == ".pdf":
        from pdfminer.high_level import extract_text

        return extract_text(str(caminho_arquivo))

    return caminho_arquivo.read_text(encoding="utf-8")
