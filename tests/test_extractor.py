"""Testes do extrator de texto."""

from __future__ import annotations

from pathlib import Path

from classificador_fiscais.extractor import extrai_texto


def test_extrai_texto_de_arquivo_txt(tmp_path: Path) -> None:
    arquivo = tmp_path / "documento.txt"
    arquivo.write_text("NOTA FISCAL ELETRONICA NF-e", encoding="utf-8")
    assert extrai_texto(arquivo) == "NOTA FISCAL ELETRONICA NF-e"


def test_extrai_texto_arquivo_inexistente_levanta_erro(tmp_path: Path) -> None:
    arquivo = tmp_path / "nao_existe.txt"
    try:
        extrai_texto(arquivo)
    except FileNotFoundError:
        return
    raise AssertionError("esperava FileNotFoundError para arquivo inexistente")
