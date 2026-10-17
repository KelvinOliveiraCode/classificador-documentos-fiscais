# BRIEF — classificador-documentos-fiscais

Projeto: classificador-documentos-fiscais
Área: IA/ML
Stack: Python, scikit-learn, TF-IDF, pdfminer.six
Saída: C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\classificador-documentos-fiscais\

## Entregas

- src/classificador_fiscais/__init__.py — exporta API pública
- src/classificador_fiscais/extractor.py — extração de texto (pdfminer.six para PDF; texto simples para demais)
- src/classificador_fiscais/classifier.py — TF-IDF + regressão logística, dataset sintético balanceado, treina/prediz/classes
- src/classificador_fiscais/pipeline.py — extração + classificação (Classificacao)
- src/classificador_fiscais/__main__.py — CLI (`python -m classificador_fiscais <arquivo>`)
- tests/test_classifier.py — dataset, treino, predição, F1 > 0,80, erros
- tests/test_extractor.py — extração de txt, arquivo inexistente
- tests/test_pipeline.py — integração extração + classificação
- pyproject.toml — metadados, dependências, scripts, config do pytest
- README.md — bilíngue PT-BR + EN (título, descrição, como usar, testes, licença MIT, autor)
- LICENSE — MIT
- BRIEF.md — este arquivo

## Critérios de aceitação

- F1 macro > 0,80
- Suíte de testes de classificação passando
- README bilíngue PT-BR + EN completo

## Comando de teste

```
python -m pytest -q
```

(executado dentro do diretório do projeto, com o ambiente `.venv` ativado)

## Resultado dos testes

9 passed. Todos verdes.
F1 macro = 1,00; acurácia = 1,00 (determinístico, random_state=42).

## Caminhos

README: C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\classificador-documentos-fiscais\README.md
BRIEF: C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\classificador-documentos-fiscais\BRIEF.md
