# Classificador de Documentos Fiscais

Classifica documentos fiscais brasileiros (nota fiscal eletrônica — NF-e e boleto bancário) a partir do texto extraído de um arquivo. Extrai texto de PDFs com pdfminer.six e treina um pipeline de TF-IDF mais regressão logística do scikit-learn para prever o tipo do documento.

MVP: extração de texto e classificação de tipo de documento.

## Stack

- Python >= 3.10
- scikit-learn >= 1.3.0
- pdfminer.six >= 20241008
- pytest >= 8.0.0 (dependência de desenvolvimento)

## Como instalar

Crie o ambiente virtual e instale as dependências:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Como usar

Execute a CLI passando um arquivo (PDF ou texto):

```
python -m classificador_fiscais documento.txt
```

Saída de exemplo:

```
Arquivo: documento.txt
Tipo previsto: nfe
```

O CLI aceita arquivos PDF (extração com pdfminer.six) e arquivos de texto simples.

## Testes

Rode a suíte pytest:

```
python -m pytest -q
```

Critério de aceitação: F1 macro maior que 0,80 e todos os testes verdes. Resultado atual: 9 testes passando, F1 macro = 1,00, acurácia = 1,00.

## Estrutura do projeto

```
src/classificador_fiscais/
  __init__.py    — API pública
  __main__.py    — CLI (python -m classificador_fiscais)
  classifier.py  — TF-IDF + regressão logística, dataset sintético, treina/prediz
  extractor.py   — extração de texto (PDF com pdfminer.six, texto simples)
  pipeline.py    — extração e classificação (Classificacao, classifica_arquivo)
tests/
  test_classifier.py
  test_extractor.py
  test_pipeline.py
pyproject.toml   — metadados, dependências, scripts, config do pytest
```

## Licença e autor

MIT. Autor: Kelvin Oliveira.
