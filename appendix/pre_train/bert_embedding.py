import os
file_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, file_path + '../../pyfunctor')
import csv_handler as csv_handler
import transform as transformer
import csv

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from gensim.utils import simple_preprocess
class BertEmbedding:
    def __init__(self, max_length = 128):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def project(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > (self.max_length - 2):
            tokens = tokens[:self.max_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        result = self.model(torch.tensor([token_ids]), output_all_encoded_layers=False)
        output = result[0][0][0].detach().numpy()
        return output


def test_bertembedding():    
    embedder = BertEmbedding()
    text = 'hello world'
    print(embedder.project(text))


#test_bertembedding()
