set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set SQUAD_DIR=squad11
set SQUAD_LAPTOP_DIR=tmp\squad2_base

python run_squad.py ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%SQUAD_LAPTOP_DIR%\model.ckpt-2350 ^
  --do_predict=True ^
  --predict_file=%SQUAD_DIR%\dev-v1.1-mjs.json ^
  --max_seq_length=384 ^
  --doc_stride=128 ^
  --output_dir=%SQUAD_LAPTOP_DIR%\ ^
  --version_2_with_negative=True