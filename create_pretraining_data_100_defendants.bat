set BERT_BASE_DIR=models\uncased_L-12_H-768_A-12

python create_pretraining_data.py ^
  --input_file=.\sample_text_100_defendants.txt ^
  --output_file=tmp\tf_examples.100defs.tfrecord ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --do_lower_case=True ^
  --max_seq_length=128 ^
  --max_predictions_per_seq=20 ^
  --masked_lm_prob=0.15 ^
  --random_seed=12345 ^
  --dupe_factor=5