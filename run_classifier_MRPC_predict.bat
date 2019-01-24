set BERT_BASE_DIR=C:\Users\michaels\Projects\bert\uncased_L-12_H-768_A-12
set GLUE_DIR=C:\Users\michaels\Projects\bert\glue_data
set TRAINED_CLASSIFIER=tmp\mrpc_output
set OUTPUT_DIR=tmp\mrpc_output_predict\

python run_classifier.py ^
  --task_name=MRPC ^
  --do_predict=true ^
  --data_dir=%GLUE_DIR%\MRPC ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%TRAINED_CLASSIFIER% ^
  --max_seq_length=128 ^
  --output_dir=%OUTPUT_DIR%
