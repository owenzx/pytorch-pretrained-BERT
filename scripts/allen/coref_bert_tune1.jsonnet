// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  "dataset_reader": {
    "type": "my_coref",
    "bert_model_name": "bert-base-uncased",
    "max_pieces": 512,
    "max_span_width": 15
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "my_coref",
    "bert_model": "bert-base-uncased",
    "bert_feedforward":{
      "input_dim": 768,
      "num_layers": 1,
      "hidden_dims": 400,
      "activations": "relu",
      "dropout": 0.2
    },
    "mention_feedforward": {
        "input_dim": 1220,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": 3680,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "lexical_dropout": 0.1,
    "feature_size": 20,
    "max_span_width": 15,
    "spans_per_word": 0.4,
    "max_antecedents": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 20,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 20,
      "num_steps_per_epoch":4738
    },
    "optimizer": {
      "type": "bert_adam",
      "lr": 2e-5,
      "t_total": -1,
      "max_grad_norm": 1.0,
    }
  }
}
