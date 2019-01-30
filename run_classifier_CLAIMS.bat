set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12
set DATA_DIR=data\all_defendants_data
set OUTPUT_DIR=tmp\claims_output\
set PRETRAINING_DIR=models\pretraining_output_all_defs

python run_classifier.py ^
  --task_name=CLAIMS ^
  --do_train=true ^
  --do_eval=true ^
  --data_dir=%DATA_DIR% ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%PRETRAINING_DIR%\model.ckpt-1000 ^
  --max_seq_length=128 ^
  --train_batch_size=32 ^
  --learning_rate=2e-5 ^
  --num_train_epochs=3.0 ^
  --output_dir=%OUTPUT_DIR%