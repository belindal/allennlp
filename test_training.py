from typing import Iterator, List, Dict
import shutil
import tempfile

import torch
import numpy as np
import allennlp

from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors.coref import CorefPredictor
from allennlp.data.dataset_readers.coreference_resolution import ConllCorefReader
from allennlp.predictors.predictor import Predictor
from allennlp.training.metrics import CategoricalAccuracy
import os

torch.manual_seed(1)

COREF_TRAIN_DATA_PATH = "../../data/coref_ontonotes/train.english.v4_gold_conll"
COREF_DEV_DATA_PATH = "../../data/coref_ontonotes/dev.english.v4_gold_conll"
COREF_TEST_DATA_PATH = "../../data/coref_ontonotes/test.english.v4_gold_conll"

# In practice you'd probably do this from the command line:
#   $ allennlp train tutorials/tagger/experiment.jsonnet -s /tmp/serialization_dir
#
def main():

    '''
    params = Params.from_file('training_config/coref.jsonnet')
    serialization_dir = "../../models" # tempfile.mkdtemp()
    # model = train_model(params, serialization_dir)
    model = Model.load(params, serialization_dir, os.path.join(serialization_dir, "weights.th"))

    # Make prediction
    predictor = CorefPredictor(model)
    '''
    # Make predictions
    predictor = Predictor.from_path("../../models")
    # '''
    docs = [{"document": "The woman reading a newspaper sat on the bench with her dog."},
            {"document": "The man looked at himself."}]
    output = predictor.predict_batch_json(
        inputs=docs,
    )
    print("output: ")
    for i in range(len(docs)):
        for item in output[i]:
            print(str(item) + ": " + str(output[i][item]))
        print()

    # '''


if __name__ == "__main__":
    main()
