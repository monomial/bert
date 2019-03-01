set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set SQUAD_OUTPUT_DIR=tmp\squad2_base
set THRESH=-6.0

python flask_main.py ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%SQUAD_OUTPUT_DIR%\model.ckpt-10859 ^
  --max_seq_length=384 ^
  --doc_stride=128 ^
  --output_dir=%SQUAD_OUTPUT_DIR%\ ^
  --version_2_with_negative=True ^
  --null_score_diff_threshold=%THRESH% ^
  --use_predictor=True