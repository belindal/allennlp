// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
{
  // Separate dataset reader for holding out labels in training data as user labels
  "dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters"
      }
    },
    "max_span_width": 10,
    "simulate_user_inputs": false,  // Sampled from training data
    "fully_labelled_threshold": 700,
  },
  "validation_dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters"
      }
    },
    "max_span_width": 10,
  },
  //"fully_labelled_train_data_path": "../data/coref_ontonotes/sample_train_full",
  //"held_out_train_data_path": "../data/coref_ontonotes/sample_train_held",
  "train_data_path": "../data/coref_ontonotes/train.english.v4_gold_conll",
  "validation_data_path": "../data/coref_ontonotes/dev.english.v4_gold_conll",
  "test_data_path": "../data/coref_ontonotes/test.english.v4_gold_conll",
  "model": {
    "type": "coref",
    "coarse_to_fine_pruning": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
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
        "num_layers": 1,
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
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ],
    "lexical_dropout": 0.5,
    "feature_size": 20,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    "max_antecedents": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "held_out_iterator": {
    "type": "basic",
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 150, //60,
    "grad_norm": 5.0,
    "patience" : 5,
    "cuda_device" : 1,
    "validation_metric": "+coref_f1",
    /*
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    */
    "optimizer": {
      "type": "adam"
    },
    "active_learning": {
      "model_type": "coref",
      "epoch_interval": 20,
      "num_labels": 5,
      "simulate_user_inputs": false, // have to update in 2 places
    }
  }
}
