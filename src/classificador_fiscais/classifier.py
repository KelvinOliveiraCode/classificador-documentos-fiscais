"""Classificador de tipo de documento fiscal (NF-e vs boleto) com TF-IDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

TIPO_NFE = "nfe"
TIPO_BOLETO = "boleto"


@dataclass(frozen=True)
class ResultadoAvaliacao:
    acuracia: float
    f1_macro: float


def dataset_exemplo() -> tuple[list[str], list[str]]:
    """Dataset sintético e determinístico de documentos fiscais.

    Cada amostra reúne várias palavras-chave da sua classe (ex.: "nota
    fiscal", "chave de acesso", "danfe", "icms" para NF-e; "boleto", "linha
    digitavel", "nosso numero", "agencia" para boleto), repetidas entre
    amostras da mesma classe. Assim treino e teste compartilham tokens
    distintivos e a separação entre classes é robusta e reproduzível. As
    classes são balanceadas.
    """
    textos_nfe = [
        "nota fiscal eletronica nf-e emitente cnpj chave de acesso danfe",
        "nota fiscal eletronica produto valor total icms serie numero",
        "chave de acesso nf-e emitente destinatario data de emissao",
        "danfe documento auxiliar nota fiscal eletronica cfop",
        "nota fiscal servico prestado iss retido valor total",
        "emitente cnpj destinatario nota fiscal eletronica nf-e",
        "numero da nota fiscal serie valor total da nota fiscal",
        "nota fiscal eletronica emitida chave de acesso protocolo autorizacao",
        "base de calculo icms aliquota produto nota fiscal nf-e",
        "nf-e danfe emitente cnpj valor total data emissao destinatario",
        "nota fiscal eletronica cfop iss retido chave de acesso",
        "danfe numero serie emitente cnpj nota fiscal eletronica",
    ]
    textos_boleto = [
        "boleto bancario carteira nosso numero agencia conta",
        "linha digitavel codigo de barras boleto vencimento valor",
        "boleto registrado nosso numero cedente sacado documento",
        "agencia conta boleto bancario pagamento vencimento",
        "codigo de barras linha digitavel boleto banco do brasil",
        "nosso numero boleto carteira agencia cobradora valor documento",
        "boleto vencimento sacado codigo cedente linha digitavel",
        "boleto bancario registrado desconto pagamento antecipado",
        "agencia e conta nosso numero boleto vencimento outubro",
        "boleto registrado banco compensacao codigo de barras documento",
        "linha digitavel nosso numero agencia conta boleto",
        "boleto vencimento valor documento sacado cedente banco",
    ]
    textos: list[str] = []
    tipos: list[str] = []
    for texto in textos_nfe:
        textos.append(texto)
        tipos.append(TIPO_NFE)
    for texto in textos_boleto:
        textos.append(texto)
        tipos.append(TIPO_BOLETO)
    return textos, tipos


class ClassificadorDocumentosFiscais:
    """Classifica o tipo de documento fiscal a partir do texto extraído."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._classes: list[str] | None = None

    def treina(self, textos: Iterable[str], tipos: Iterable[str]) -> ResultadoAvaliacao:
        """Treina o modelo e retorna métricas de validação (hold-out)."""
        lista_textos = list(textos)
        lista_tipos = list(tipos)
        if len(lista_textos) != len(lista_tipos):
            raise ValueError("textos e tipos devem ter o mesmo tamanho.")
        if not lista_textos:
            raise ValueError("O dataset de treinamento não pode estar vazio.")

        self._classes = sorted(set(lista_tipos))

        treino_textos, teste_textos, treino_tipos, teste_tipos = train_test_split(
            lista_textos,
            lista_tipos,
            test_size=0.25,
            random_state=self.random_state,
            stratify=lista_tipos,
        )

        self._pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(lowercase=True)),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                ),
            ]
        )
        self._pipeline.fit(treino_textos, treino_tipos)

        predicoes = self._pipeline.predict(teste_textos)
        return ResultadoAvaliacao(
            acuracia=float(self._pipeline.score(teste_textos, teste_tipos)),
            f1_macro=float(f1_score(teste_tipos, predicoes, average="macro")),
        )

    def prediz(self, texto: str) -> str:
        """Retorna o tipo previsto para um texto."""
        if self._pipeline is None:
            raise RuntimeError("O modelo ainda não foi treinado. Chame treina().")
        return str(self._pipeline.predict([texto])[0])

    def classes(self) -> list[str]:
        """Retorna as classes aprendidas."""
        if self._classes is None:
            raise RuntimeError("O modelo ainda não foi treinado.")
        return list(self._classes)
