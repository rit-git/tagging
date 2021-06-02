import sys
import os

if 'FACTMINE_HOME' in os.environ:
    pyfunctor_path = os.path.join(os.environ['FACTMINE_HOME'], 'third/pyfunctor')
    sys.path.append(pyfunctor_path)
else:
    sys.exit("please declara environment variable 'FACTMINE_HOME'")
import csv_handler as csv_handler
import transform as transform
import nlp.util as util

import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers.data.processors.glue import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange

from sklearn.metrics import precision_recall_fscore_support

import numpy

class BertModel:
    def __init__(self, model_dir = "bert-base-uncased", max_seq_length = 128, from_tf=False):
        # define model
        #if model_dir == "bert-base-uncased": # if it is a untrained model
        config = BertConfig.from_pretrained(
            model_dir,
            num_labels = 2,
            finetuning_task = 'SST-2',
            cache_dir = None
        )

        self.model = BertForSequenceClassification.from_pretrained(
            model_dir,
            from_tf = from_tf,
            config = config,
            cache_dir = None,
        )
        #else: # if it is a well-trained  model 
        #    self.model = BertForSequenceClassification.from_pretrained(model_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True) 
        self.max_seq_length = max_seq_length
        self.dummy_label = "-1" # dummy label
        self.output_mode = "classification"

    # output: score_0, score_1, pred_argmax_class, text
    def predict(self, all_texts, batch_size = 32, text_col = 1):

        if type(all_texts[0]) == tuple or type(all_texts[0]) == list:
            all_texts = transform.map_func(all_texts, lambda row : row[text_col])

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

    # train BERT model with labels
    def train(self, labeled_dataset, train_batch_size = 32, num_epoch = 5, adam_lr = 2e-5, adam_epsilon = 1e-8, scheduler_warmup_steps = 0, text_col=1, label_col = 2):

        # prepare training data
        texts = transform.map_func(labeled_dataset, lambda tri : tri[text_col])

        labels = transform.map_func(labeled_dataset, lambda tri : tri[label_col])

        train_examples = self.__get_examples(texts, labels)

        train_dataset = self.__get_inputs(train_examples, ["0", "1"])

        train_dataset = TensorDataset(train_dataset['input_ids'], train_dataset['attention_mask'], train_dataset['token_type_ids'], train_dataset['labels'])
        
        train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        # prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr = adam_lr, eps = adam_epsilon)

        # prepare scheduler
        t_total = len(train_dataloader) * num_epoch

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps = t_total 
        )

        # start training
        self.model.zero_grad()

        for _ in trange(0, num_epoch, desc = "Training Epoch"):
            num_step_per_epoch = len(train_dataloader)
            for step, batch in enumerate(tqdm(train_dataloader, desc = "Iteration")):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = self.model(**inputs)
                loss = outputs[0]
                if step in {int(num_step_per_epoch/4),int(num_step_per_epoch * 2/4),int(num_step_per_epoch * 3/4),num_step_per_epoch - 1} : 
                    print("\n training loss is " + str(loss.item()))
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

        # return trained model
        return self.model

    def evaluate(self, dev_dataset):
        preds = self.predict(dev_dataset)
        ground = transform.map_func(dev_dataset, lambda row : int(row[2]))
        (precision, recall, fscore, support) = precision_recall_fscore_support(ground, preds)
        return fscore

    def checkpoint(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    
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
        
    # convert a BERT input text to BERT input sequence
    def __get_inputs(self, examples, label_list = None):
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=[self.dummy_label] if label_list == None else label_list,
            max_length=self.max_seq_length,
            output_mode=self.output_mode,
#pad_on_left=False,
#pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
#            pad_token_segment_id=0
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

    # convert a text to BERT Input text
    def __get_examples(self, batch, labels = None):
        examples = []
        for (i, txt) in enumerate(batch):
            guid = "%s" % (i)
            text_a = txt 
            label = labels[i] if labels != None else self.dummy_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        

def bert_estimate(input_path, text_col, output_path, model_dir, gpu, with_header):

    # assign GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # load model
    if model_dir == '':
        model_dir = os.path.join(os.environ['FACTMINE_HOME'], 'cache/company_model') 

    print("Loading model from %s ...\n" % (model_dir))

    model = BertModel(model_dir)

    print("Calculating negative score, positive score, and argmax class...\n")

    dataset = csv_handler.csv_readlines(input_path)
    header = None
    if with_header == True:
        header = dataset[0]
        dataset = dataset[1:]

    text_col = text_col - 1
    texts = transform.map_func(dataset, lambda row : row[text_col])

    preds = model.predict(texts)

    assert(len(dataset) == len(preds))

    # apply sigmoid
    preds = transform.map_func(preds, lambda quad : [util.sigmoid(quad[0]), util.sigmoid(quad[1]), quad[2], quad[3]])

    for i in range(len(dataset)):
        dataset[i].append(preds[i][0])
        dataset[i].append(preds[i][1])
        dataset[i].append(preds[i][2])

    if with_header == True:
        header.append('score_0')
        header.append('score_1')
        header.append('arg_class')
        dataset = [header] + dataset

    csv_handler.csv_writelines(output_path, dataset)

    if output_path != "":
        print("Finished, Results are ready at %s " % (output_path))

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
    
