set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set SQUAD_DIR=squad11
set SQUAD_OUTPUT_DIR=tmp\squad_base

python %SQUAD_DIR%\evaluate-v1.1.py %SQUAD_DIR%\dev-v1.1.json %SQUAD_OUTPUT_DIR%\predictions.json