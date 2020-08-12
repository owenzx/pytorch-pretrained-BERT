// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
//   + BERT

local bert_model = "bert-base-uncased";
local max_length = 128;
local feature_size = 20;
local max_span_width = 25;

local bert_dim = 768;  # uniquely determined by bert_model
local lstm_dim = 768;
local span_embedding_dim = 3 * bert_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

{
  "dataset_reader": {
    "type": "coref_head_joint_bert_subword",
    "wordpiece_modeling_tokenizer":{
      "model_name": bert_model,
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": bert_model,
        "max_length": max_length
      },
    },
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "coref_head_joint_bert_attention_cheap_light",
    "text_field_embedder": {
      "type": "many",
      "token_embedders": {
        "tokens": {
            "type": "attention_pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
        }
      }
    },
    "token_mask_feedforward": {
      "input_dim": 400,
      "num_layers": 1,
      "hidden_dims": 1,
      "activations": "linear",
    },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 1
    },

    "initializer": {"regexes":[
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ]},
    "heads_per_word": 0.5,
    "lexical_dropout": 0.2,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 20,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  }
}