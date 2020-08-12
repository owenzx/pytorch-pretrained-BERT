// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
{
  "dataset_reader": {
    "type": "h2s",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens":  true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        },
        "min_padding_length": 5
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path":  std.extVar("DEV_DATA_PATH"),
  "model": {
    "type": "from_archive",
    "archive_file": std.extVar("LOAD_MODEL_PATH"),
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 40
    }
  },

  "trainer": {
    "num_epochs": 100,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+span_acc",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}