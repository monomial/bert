# for the flask web server serving
from flask import Flask
from flask import request
from flask import jsonify

import collections
import json
import math
import os
import random

import modeling
import optimization
import tokenization

import tensorflow as tf

# for the SQuAD 2.0 code
import run_squad

flags = tf.flags

FLAGS = flags.FLAGS

app = Flask(__name__)

def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

estimator = None
fullTokenizer = None

def initEstimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    fullTokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    
    run_config = tf.contrib.tpu.RunConfig(
      cluster=None,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
    
    model_fn = run_squad.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=None,
      num_warmup_steps=None,
      use_tpu=False,
      use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

    return estimator


# run the estimator init
# so that our estimator is set up
initEstimator()

@app.route("/")
def hello():
    initEstimator()
    return "Hello World!"

@app.route("/predict", methods=['POST', 'GET'])
def runTest():

    estimator = initEstimator()

    post = request.json

    #if post != None:
    #    return jsonify(post)

    paragraph_text = post['text']
    question_text = post['question']
    
    char_to_word_offset = []
    doc_tokens = run_squad.convert_paragraph_text_to_doc_tokens(paragraph_text, char_to_word_offset)

    example = run_squad.SquadExample(
        qas_id="999",
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=False)

    #return jsonify(str(example))

    eval_examples = [example]

    eval_writer = run_squad.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    fullTokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
    
    run_squad.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=fullTokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = run_squad.input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          run_squad.RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    run_squad.write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)

    return jsonify(all_results)


if __name__ == "__main__":
    app.run(debug=True)