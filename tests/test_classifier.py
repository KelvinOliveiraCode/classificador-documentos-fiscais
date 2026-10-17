"""Testes do classificador de documentos fiscais."""

from __future__ import annotations

from classificador_fiscais.classifier import (
    ClassificadorDocumentosFiscais,
    TIPO_BOLETO,
    TIPO_NFE,
    dataset_exemplo,
)


def test_dataset_exemplo_retorna_textos_e_tipos_balanceados() -> None:
    textos, tipos = dataset_exemplo()
    assert len(textos) == len(tipos)
    assert len(textos) > 0
    assert set(tipos) == {TIPO_NFE, TIPO_BOLETO}
    assert tipos.count(TIPO_NFE) == tipos.count(TIPO_BOLETO)


def test_treina_aprende_classes_e_prediz_acerca_nfe() -> None:
    textos, tipos = dataset_exemplo()
    modelo = ClassificadorDocumentosFiscais()
    modelo.treina(textos, tipos)
    assert set(modelo.classes()) == {TIPO_NFE, TIPO_BOLETO}
    assert modelo.prediz("NOTA FISCAL ELETRONICA NF-e chave de acesso") == TIPO_NFE


def test_treina_aprende_classes_e_prediz_acerca_boleto() -> None:
    textos, tipos = dataset_exemplo()
    modelo = ClassificadorDocumentosFiscais()
    modelo.treina(textos, tipos)
    assert modelo.prediz("BOLETO BANCARIO linha digitavel nosso numero") == TIPO_BOLETO


def test_f1_macro_maior_que_0_80() -> None:
    textos, tipos = dataset_exemplo()
    modelo = ClassificadorDocumentosFiscais()
    resultado = modelo.treina(textos, tipos)
    assert resultado.f1_macro > 0.80


def test_prediz_sem_treino_levanta_erro() -> None:
    modelo = ClassificadorDocumentosFiscais()
    try:
        modelo.prediz("qualquer texto")
    except RuntimeError:
        return
    raise AssertionError("esperava RuntimeError ao predizer sem treino")


def test_treina_dataset_vazio_levanta_erro() -> None:
    modelo = ClassificadorDocumentosFiscais()
    try:
        modelo.treina([], [])
    except ValueError:
        return
    raise AssertionError("esperava ValueError para dataset vazio")
