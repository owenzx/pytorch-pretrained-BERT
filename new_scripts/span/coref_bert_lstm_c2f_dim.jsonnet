// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
//   + BERT

local bert_model = "bert-base-uncased";
local max_length = 128;
local feature_size = 20;
local max_span_width = std.parseInt(std.extVar("SPWID"));

local bert_dim = 768;  # uniquely determined by bert_model
local lstm_dim = 768;
local span_embedding_dim = 3 * bert_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

{
  "dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": bert_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width,
    "max_sentences": 60,

  },
  "validation_dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": bert_model,
        "max_length": max_length
      },
    },
    "max_span_width": max_span_width,
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "coref",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": bert_model,
            "max_length": max_length
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": bert_dim,
    },
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": std.parseInt(std.extVar("DIM")),
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": std.parseInt(std.extVar("DIM")),
        "activations": "relu",
        "dropout": 0.2
    },
    "initializer": {"regexes":[
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ]},
    "lexical_dropout": 0.5,
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "coarse_to_fine": true,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["text"],
      "padding_noise": 0.0,
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  }
}
