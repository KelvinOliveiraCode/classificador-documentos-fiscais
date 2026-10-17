# Classificador de Documentos Fiscais

Classifies Brazilian fiscal documents (electronic invoice — NF-e and bank slip) from the text extracted from a file. Extracts text from PDFs with pdfminer.six and trains a TF-IDF plus logistic regression pipeline from scikit-learn to predict the document type.

MVP: text extraction and document type classification.

## Stack

- Python >= 3.10
- scikit-learn >= 1.3.0
- pdfminer.six >= 20241008
- pytest >= 8.0.0 (dev dependency)

## Installation

Create the virtual environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

Run the CLI passing a file (PDF or text):

```
python -m classificador_fiscais document.txt
```

Example output:

```
Arquivo: document.txt
Tipo previsto: nfe
```

The CLI accepts PDF files (text extraction with pdfminer.six) and plain text files.

## Tests

Run the pytest suite:

```
python -m pytest -q
```

Acceptance criterion: macro F1 greater than 0.80 and all tests green. Current result: 9 tests passing, macro F1 = 1.00, accuracy = 1.00.

## Project structure

```
src/classificador_fiscais/
  __init__.py    — public API
  __main__.py    — CLI (python -m classificador_fiscais)
  classifier.py  — TF-IDF + logistic regression, synthetic dataset, train/predict
  extractor.py   — text extraction (PDF with pdfminer.six, plain text)
  pipeline.py    — extraction and classification (Classificacao, classifica_arquivo)
tests/
  test_classifier.py
  test_extractor.py
  test_pipeline.py
pyproject.toml   — metadata, dependencies, scripts, pytest config
```

## License and author

MIT. Author: Kelvin Oliveira.
