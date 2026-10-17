"""Classificador de documentos fiscais (NF-e / boleto)."""

from .classifier import (
    ClassificadorDocumentosFiscais,
    ResultadoAvaliacao,
    dataset_exemplo,
)
from .pipeline import Classificacao, classifica_arquivo

__all__ = [
    "ClassificadorDocumentosFiscais",
    "ResultadoAvaliacao",
    "Classificacao",
    "classifica_arquivo",
    "dataset_exemplo",
]
