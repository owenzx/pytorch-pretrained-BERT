// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  "dataset_reader": {
    "type": "coref_head_joint_bert_truncate",
    "bert_model_name": "bert-base-uncased",
    "max_pieces": 512,
    "lowercase_input": true
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "coref_head_joint_bert_truncate",
    "bert_model": "bert-base-uncased",
    "lexical_dropout": 0.1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 8
  },
  "trainer": {
    "num_epochs": 20,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 20,
      "num_steps_per_epoch":9470
    },
    "optimizer": {
      "type": "bert_adam",
      "lr": 2e-6,
      "t_total": -1,
      "max_grad_norm": 1.0,
    }
  }
}
