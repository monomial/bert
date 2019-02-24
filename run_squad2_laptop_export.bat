set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set SQUAD_DIR=squad11
set SQUAD_OUTPUT_DIR=tmp\squad2_base_laptop
set SQUAD_EXPORT_DIR=tmp\squad2_base_laptop_export

python run_squad.py ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%SQUAD_OUTPUT_DIR%\model.ckpt-10859 ^
  --do_export=True ^
  --max_seq_length=384 ^
  --doc_stride=128 ^
  --export_dir=%SQUAD_EXPORT_DIR%\ ^
  --output_dir=%SQUAD_OUTPUT_DIR%\ ^
  --version_2_with_negative=True