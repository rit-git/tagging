import os
file_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, '../../pyfunctor')
import csv_handler as csv_handler
import transform as transform
import csv

import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel

class BertEmbedding:
    def __init__(self, max_length = 512):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def project(self, text):

        tokens = self.__tokenize(text, pad_to_max_length=False) 
        token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)

        output = self.model(token_tensor)

        result = output[0][0][0].detach().cpu().numpy()
        return result

    # batch requires all texts the same length
    def project_batch(self, texts):

        batch_tokens = transform.map_func(texts, lambda text: self.__tokenize(text, pad_to_max_length=True))

        batch_tensor = torch.tensor(batch_tokens).to(self.device)

        self.model.eval()
        output = self.model(batch_tensor)

        token_embeddings = output[0].detach().cpu().numpy()

        result = transform.map_func(token_embeddings, lambda row : row[0])

        #result = output[0][0][0].detach().cpu().numpy()

        return result

    def __tokenize(self, text, pad_to_max_length=True):
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=pad_to_max_length)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [tokens[-1]]
        #if len(tokens) < self.max_length:
            
        return tokens


if __name__ == "__main__":
    embedder = BertEmbedding()

    #string = "hello"
    #for i in range(512):
    #    string += " "
    #    string += "hello"
    #result = embedder.project(string)

    batch = ['hello', 'world']
    result = embedder.project_batch(batch)

    print(result)

