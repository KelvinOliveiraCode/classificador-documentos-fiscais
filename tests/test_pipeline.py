"""Testes de integracao do pipeline (extracao + classificacao)."""

from __future__ import annotations

from pathlib import Path

from classificador_fiscais.classifier import (
    ClassificadorDocumentosFiscais,
    TIPO_NFE,
    dataset_exemplo,
)
from classificador_fiscais.pipeline import Classificacao, classifica_arquivo


def test_classifica_arquivo_retorna_tipo_e_texto(tmp_path: Path) -> None:
    arquivo = tmp_path / "documento.txt"
    arquivo.write_text("NOTA FISCAL ELETRONICA NF-e", encoding="utf-8")

    modelo = ClassificadorDocumentosFiscais()
    textos, tipos = dataset_exemplo()
    modelo.treina(textos, tipos)

    resultado = classifica_arquivo(arquivo, modelo)
    assert isinstance(resultado, Classificacao)
    assert resultado.tipo == TIPO_NFE
    assert "NF-e" in resultado.texto
