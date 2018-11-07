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

torch.manual_seed(1)


# In practice you'd probably do this from the command line:
#   $ allennlp train tutorials/tagger/experiment.jsonnet -s /tmp/serialization_dir
#
def main():


    params = Params.from_file('config/coref.jsonnet')  # Not just model parameters in here
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    
    #reader = ConllCorefReader(model, reader="")
    #reader.read()  # file paths


    # Make prediction
    predictor = CorefPredictor(model)

    ''' from sample code-- but this won't update
    # Make predictions
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
            # this won't update if I edit the code?
    '''

    predictor.predict(
        document="The woman reading a newspaper sat on the bench with her dog."
    )


if __name__ == "__main__":
    main()
