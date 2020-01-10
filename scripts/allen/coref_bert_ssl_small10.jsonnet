// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  "dataset_reader": {
    "type": "my_coref",
    "bert_model_name": "bert-base-uncased",
    "max_pieces": 512,
    "max_span_width": 25
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "unl_data_path": std.extVar("COREF_UNL_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "my_coref",
    "bert_model": "bert-base-uncased",
    "semi_supervise": true,
    "mention_dict_path": "./cache/debug_conll_train.corpus",
    "basic_switch_type": "glove_mention",
    "mask_mention": false,
    "lambda_consist": 10000.0,
    "consistency_threshold": 99999,
    "consistency_loss": true,
    "mention_feedforward": {
        "input_dim": 2324,
        "num_layers": 2,
        "hidden_dims": 600,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": 6992,
        "num_layers": 2,
        "hidden_dims": 600,
        "activations": "relu",
        "dropout": 0.2
    },
    "lexical_dropout": 0.1,
    "feature_size": 20,
    "max_span_width": 25,
    "spans_per_word": 0.4,
    "max_antecedents": 150
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "type": "ssl-trainer",
    "gradient_accumulation_steps": 2,
    "gradient_accumulation_normalization": true,
    "num_epochs": 20,
    "cuda_device" : 0,
    "step_unlabel": 1,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 20,
      "num_steps_per_epoch":491
    },
    "optimizer": {
      "type": "bert_adam",
      "lr": 2e-5,
      "t_total": -1,
      "max_grad_norm": 1.0,
      "parameter_groups":[
        [["bert_model.*bias", "bert_model.*LayerNorm.bias", "bert_model.*LayerNorm.weight", "bert_model.*layer_norm.weight"], {"weight_decay": 0.0}],
        [["_antecedent_feedforward", "_mention_pruner", "_antecedent_scorer", "_endpoint_span_extractor", "_attentive_span_extractor", "_distance_embedding"], {"lr":2e-4, "parameter_groups":[[["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}]]}]
      ]
    }
  }
}
