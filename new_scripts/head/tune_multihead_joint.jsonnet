// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  "random_seed" : 13370,
  "numpy_seed" : 1337,
  "pytorch_seed" : 133,
  "dataset_reader": {
    "type": "coref_head_joint",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      }
    },
    "max_span_width": 10
  },
  "train_data_path": std.extVar("COREF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("COREF_DEV_DATA_PATH"),
  "test_data_path": std.extVar("COREF_TEST_DATA_PATH"),
  "model": {
    "type": "coref_head_joint_mh",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [5]
            }
        }
      }
    },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 400,
        "hidden_size": 200,
        "num_layers": 1
    },
    "initializer": {"regexes":[
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ]},
    "num_heads": std.parseInt(std.extVar("NH")),
    "lexical_dropout": 0.5
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": [
        "text"
      ],
      "padding_noise": 0.0,
      "batch_size": 8,
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
      "type": "adam"
    }
  }
}
