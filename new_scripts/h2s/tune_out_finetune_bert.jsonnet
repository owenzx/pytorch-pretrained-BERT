// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
local bert_model = "bert-base-uncased";
local epochs = std.parseInt(std.extVar("EPC"));
{
 "dataset_reader": {
    "type": "h2s_bert",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": bert_model,
      },
    },
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
      "batch_size": 5
    }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": std.parseJson(std.extVar("LR")),
      "eps": 1e-8
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
    "cuda_device": 0,
    "validation_metric": "+span_acc",
  }
}