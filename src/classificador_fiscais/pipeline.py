"""Pipeline: extracao de texto seguida de classificacao."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .classifier import ClassificadorDocumentosFiscais
from .extractor import extrai_texto


@dataclass(frozen=True)
class Classificacao:
    tipo: str
    texto: str


def classifica_arquivo(
    caminho: str | Path, classificador: ClassificadorDocumentosFiscais
) -> Classificacao:
    """Extrai o texto do arquivo e classifica o tipo do documento fiscal."""
    texto = extrai_texto(caminho)
    tipo = classificador.prediz(texto)
    return Classificacao(tipo=tipo, texto=texto)
