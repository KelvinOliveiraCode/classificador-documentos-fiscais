# classificador-documentos-fiscais

**English** · [Português (Brasil)](README.md)

[![Python 3](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn)](#) [![pdfminer](https://img.shields.io/badge/pdfminer.six-2024%2B-3776AB?style=flat-square)](#) [![pytest](https://img.shields.io/badge/pytest-9%20tests-0A9EDC?style=flat-square&logo=pytest)](#) [![MIT License](https://img.shields.io/badge/License-MIT-f4c430?style=flat-square)](LICENSE)

## What it is

**A Brazilian fiscal-document classifier**: it extracts text from a PDF/file (via pdfminer.six), trains a **text-classification model** (TF-IDF + Logistic Regression) on a sample dataset and predicts the document type (e.g. NF-e, boleto). It is an educational ML lab applied to documents.

## Status

An educational lab, working and tested (9 tests).

## Features

- **Text extraction** (`extractor.py`): `extrai_texto(path)` reads a PDF (pdfminer) or text file.
- **Classifier** (`classifier.py`): `ClassificadorDocumentosFiscais` with `treina`, `prediz`, `classes`; a TF-IDF + LogisticRegression pipeline (`random_state` configurable, default 42).
- **Evaluation**: accuracy and macro-F1.
- **Pipeline** (`pipeline.py`): `classifica_arquivo` — extracts, classifies and returns a `Classificacao`.
- **CLI**: `classificador-fiscais <file.pdf|txt>` prints the predicted type.

## Architecture

A `classificador_fiscais` package with `extractor` (extraction), `classifier` (TF-IDF+LR model) and `pipeline` (orchestration), wired to the CLI.

```mermaid
flowchart LR
    DOC[PDF/file] --> EXT[extractor - extract text]
    EXT --> PIP[pipeline - classifica_arquivo]
    PIP --> CLF[classifier - TF-IDF + LogisticRegression]
    CLF --> TYPE[predicted type: NF-e/boleto]
    CLI[classificador-fiscais] --> EXT
```

## Stack

- Python ≥ 3.10
- scikit-learn ≥ 1.3, pdfminer.six ≥ 2024
- pytest (dev)

## Repository layout

```
classificador-documentos-fiscais/
├── pyproject.toml · BRIEF.md
├── src/classificador_fiscais/
│   ├── extractor.py   # text extraction (pdfminer)
│   ├── classifier.py  # TF-IDF + LR model
│   ├── pipeline.py    # orchestration
│   └── __main__.py    # CLI
├── tests/             # 9 tests
└── LICENSE            # MIT
```

## Prerequisites and installation

```bash
pip install -e .
```

## How to run

```bash
classificador-fiscais nf-e.pdf
classificador-fiscais boleto.txt
```

## Tests and quality

```bash
PYTHONPATH=src python -m pytest tests/
# 9 passed
```

## Technical decisions and trade-offs

- **TF-IDF + Logistic Regression**: a classic, light text-classification model; interpretable and fast to train.
- **Embedded sample dataset**: runs and tests without external data; the trade-off is that production use requires real data.
- **Accuracy/F1-macro**: balanced multiclass classification metrics.

## Known limitations

- Trained on a small sample dataset (limited types: NF-e, boleto); not a full fiscal classifier.
- Text extraction depends on PDF quality/OCR (no built-in OCR).
- The trained model is not persisted (retrains on each run).

## Development process

An individual project by Kelvin Oliveira, built with orchestrated AI agents and reviewed by the author.

## License and author

MIT — Copyright (c) 2026 Kelvin Rebelo de Oliveira. Author: [Kelvin Oliveira](https://github.com/KelvinOliveiraCode) · [Portfolio](https://kelvinoliveiracode.github.io/portfolio/)
