# Classificador de Documentos Fiscais

## pt-BR

### O que faz

Classifica documentos fiscais (nota fiscal eletrônica — NF-e e boleto bancário) a partir do texto extraído do arquivo. Extrai texto de PDFs com pdfminer.six e treina um modelo de TF-IDF mais regressão logística do scikit-learn para prever o tipo do documento.

MVP: extração de texto + classificação de tipo de documento.

Stack: Python, scikit-learn, TF-IDF, pdfminer.six.

### Como usar

Crie o ambiente e instale as dependências:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Execute a CLI passando um arquivo (PDF ou texto):

```
python -m classificador_fiscais documento.txt
```

Saída de exemplo:

```
Arquivo: documento.txt
Tipo previsto: nfe
```

### Testes

Rode a suíte pytest:

```
python -m pytest -q
```

Critério de aceitação: F1 macro maior que 0,80 e todos os testes verdes.

### Licença e autor

MIT. Autor: Kelvin Oliveira.

## en

### What it does

Classifies fiscal documents (electronic invoice — NF-e and bank slip) from the text extracted from a file. Extracts text from PDFs with pdfminer.six and trains a TF-IDF plus logistic regression model from scikit-learn to predict the document type.

MVP: text extraction + document type classification.

Stack: Python, scikit-learn, TF-IDF, pdfminer.six.

### How to use

Create the environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Run the CLI with a file (PDF or text):

```
python -m classificador_fiscais document.txt
```

Example output:

```
Arquivo: document.txt
Tipo previsto: nfe
```

### Tests

Run the pytest suite:

```
python -m pytest -q
```

Acceptance criterion: macro F1 greater than 0.80 and all tests green.

### License and author

MIT. Author: Kelvin Oliveira.
