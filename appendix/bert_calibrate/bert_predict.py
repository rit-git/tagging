import torch
import sys
sys.path.insert(0, "../../pyfunctor")
import csv_handler as csv_handler
import transform as transform

from transformers import BertForSequenceClassification, BertTokenizer
from transformers.data.processors.glue import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import numpy

class BertModel:
    def __init__(self, model_dir, max_seq_length = 128):
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True) 
        self.max_seq_length = max_seq_length
        self.dummy_label = "-1" # dummy label
        self.output_mode = "classification"

    # output: score_0, score_1, pred_class, text
    def predict(self, all_texts, batch_size = 32):
        output = []
        i = 0

        while i < len(all_texts):
            print("start processing {} / {}".format(i, len(all_texts)))
            batch_texts = all_texts[i:(i + batch_size)] 
            examples = self.__get_examples(batch_texts)
            pred = self.__predict_batch(examples)
            pred_and_class = self.__get_class_from_pred(pred)

            assert(len(pred_and_class) == len(batch_texts))
            transform.map_func(range(len(pred_and_class)), lambda idx : pred_and_class[idx].append(batch_texts[idx]))

            output += pred_and_class

            i += batch_size

        return output

    def __get_class_from_pred(self, pred):
        result = pred[1]
        result = result.detach().cpu().numpy()

        label = numpy.argmax(result, axis = 1)

        result = result.tolist()
        label = label.tolist()

        transform.map_func(range(len(result)), lambda idx : result[idx].append(label[idx]))

        return result
        
    def __predict_batch(self, examples):
        self.model.eval()
        with torch.no_grad():
            inputs = self.__get_inputs(examples)
            pred = self.model(**inputs)
        return pred
        
    def __get_inputs(self, examples):
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=[self.dummy_label],
            max_length=self.max_seq_length,
            output_mode=self.output_mode,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features],dtype=torch.long)

        all_input_ids = all_input_ids.to(self.device)
        all_attention_mask = all_attention_mask.to(self.device)
        all_token_type_ids = all_token_type_ids.to(self.device)
        all_labels = all_labels.to(self.device)

        inputs = {"input_ids":all_input_ids, "attention_mask":all_attention_mask, "token_type_ids":all_token_type_ids, "labels": all_labels}
        return inputs 

    def __get_examples(self, batch):
        examples = []
        for (i, txt) in enumerate(batch):
            guid = "%s" % (i)
            text_a = txt 
            label = self.dummy_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        

if __name__ == "__main__":

    data_path = sys.argv[1]
    model_dir = sys.argv[2]
    output_path = sys.argv[3]

    # load test dataset
    raw_dataset = csv_handler.csv_readlines(data_path)
    ids = transform.map_func(raw_dataset, lambda row : row[0])
    texts = transform.map_func(raw_dataset, lambda row : row[1])

    # load model
    model = BertModel(model_dir)

    pred = model.predict(texts, 100)

    assert(len(ids) == len(pred))
    output = transform.map_func(range(len(ids)), lambda idx : [ids[idx]] + pred[idx])
    csv_handler.csv_writelines(output_path, output)
    
