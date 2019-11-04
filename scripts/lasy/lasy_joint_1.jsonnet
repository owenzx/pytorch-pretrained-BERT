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
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "my_coref_lasy",
    "bert_model": "bert-base-uncased",
    "consistency_loss": true,
    "lambda_consist": 10000.0,
    "mention_switcher_type": "controller",
    "mention_dict_path": "./cache/debug_conll_train.corpus",
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
    "max_antecedents": 150,
    "ways_arg": ["switch_glove", "switch_pron", "add_clause", "simplify_np"],
    "num_aug_prob": 5,
    "num_aug_magn": 4,
    "controller_hid": 128,
    "softmax_temperature": 1.0,
    "num_mix": 3,
    "input_aware": false,
    "baseline": "greedy",
    "initializer":[
      ["(?!^mention_switcher.*$)(^.*$)", {"type": "pretrained", "weights_file_path": "./outputs/mentionswitch_really_baseline_0917/best.th", "parameter_name_overrides": {}}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "train_model":true,
    "train_controller":true,
    "type": "lasy-trainer",
    "num_epochs": 20,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 20,
      "num_steps_per_epoch":4735
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
    },
    "optimizer_controller": {
      "type": "adam",
      "lr": 1e-5
    }
  }
}
