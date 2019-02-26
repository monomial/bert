set SQUAD_DIR=squad11
set SQUAD_OUTPUT_DIR=tmp\squad2_base

python %SQUAD_DIR%\evaluate-v2.0.py ^
       %SQUAD_DIR%\dev-v2.0.json ^
       %SQUAD_OUTPUT_DIR%\predictions.json ^
       --na-prob-file %SQUAD_OUTPUT_DIR%\null_odds.json