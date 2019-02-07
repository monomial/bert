set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set SQUAD_DIR=squad11
set SQUAD_LAPTOP_DIR=tmp\squad_base_laptop

python run_squad.py ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%SQUAD_LAPTOP_DIR%\model.ckpt-8000 ^
  --do_predict=True ^
  --predict_file=%SQUAD_DIR%\dev-v1.1.json ^
  --train_batch_size=12 ^
  --learning_rate=3e-5 ^
  --num_train_epochs=2.0 ^
  --max_seq_length=384 ^
  --doc_stride=128 ^
  --output_dir=%SQUAD_LAPTOP_DIR%\