import argparse
import os
import torch
import torch.nn as nn

from torchtext.data import TabularDataset, BucketIterator
from torchtext.data import Field
from torchtext.vocab import Vectors, GloVe
from tqdm import tqdm, trange
import sys
import os
sys.path.insert(0, "../../pyfunctor")
sys.path.insert(0, "../../model")
from cnn import CNNModel
from lstm import LSTMModel
from bilstm import BILSTMModel

from sklearn import metrics
import csv_handler as csv_handler
import transform as transform
import time
#from util.weight import WeightClassCSV 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path, batch_size, max_seq_length, glove="840B", emb_size=300):
	TEXT = Field(sequential=True, fix_length=max_seq_length, lower=True)
	LABEL = Field(sequential=False, use_vocab=False)
	ID = Field(sequential=False, use_vocab=False)

	data_fields = [("id", ID), 
				   ("sent", TEXT),
				   ("label", LABEL)]
	train_path = os.path.join(path, "train.csv")
	train = TabularDataset(path=train_path, format="csv", skip_header=False,
		fields=data_fields)
	test_path = os.path.join(path, "dev.csv")
	test = TabularDataset(path=test_path, format="csv", skip_header=False,
		fields=data_fields)

	TEXT.build_vocab(train, vectors=GloVe(name=glove, dim=emb_size))
	LABEL.build_vocab(train)

	vocab_size = len(TEXT.vocab)
	vocab_weights = TEXT.vocab.vectors

	train_iter = BucketIterator(dataset=train, batch_size=batch_size,
		sort_key=lambda x: x.id, shuffle=True, repeat=False)
	test_iter = BucketIterator(dataset=test, batch_size=batch_size,
		sort_key=lambda x: x.id, shuffle=False, repeat=False)

	return train_iter, test_iter, vocab_size, vocab_weights

def F1(predicts, golds):
	true_predict = 0
	true = 0
	predict = 0

	for i in range(len(predicts)):
		if predicts[i] == 1:
			predict += 1
		if golds[i] == 1:
			true += 1
		if predicts[i] == 1 and golds[i] == 1:
			true_predict += 1

	precision = (true_predict+0.0)/(predict+0.0) if predict>0 else 0
	recall = (true_predict+0.0)/(true+0.0) if true>0 else 0
	f1 = (2*precision*recall)/(precision+recall) if predict>0 and true>0 else 0

	return precision, recall, f1


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                                            default=None,
                    type=str,
                    required=True,
                    help="Dataset folder")
    parser.add_argument("--model",
                                            default=None,
                    type=str,
                    required=True,
                    help="Model type: CNN, LSTM or BILSTM")
    parser.add_argument("--glove",
                                            default="840B",
                    type=str,
                    help="Golve version (6B, 42B, 840B)")
    parser.add_argument("--emb_size",
                                            default=300,
                    type=int,
                    help="Golve embedding size (100, 200, 300)")
    parser.add_argument("--max_seq_length",
                                            default=256,
                    type=int,
                    help="Maximum sequence length")
    parser.add_argument("--num_epoch",
                                            default=9,
                    type=int,
                    help="Number of training epoch")
    parser.add_argument("--batch_size",
                                            default=32,
                    type=int,
                    help="Batch size")
    parser.add_argument("--lr",
                                            default=1e-4,
                    type=float,
                    help="Learning rate")
    parser.add_argument("--fix_emb",
                                            default=False,
                    type=bool,
                    help="Fix embedding layer")

    parser.add_argument("--log_file",
                                            default=False,
                    type=str,
                    required=True,
                    help="log file path")

    args = parser.parse_args()
    
    # Load data
    print("Loading data ...")
    train_iter, test_iter, vocab_size, vocab_weights = load_data(args.dataset, 
            args.batch_size, args.max_seq_length, glove=args.glove, emb_size=args.emb_size)

    # Initialize model
    assert args.model in ["CNN", "LSTM", "BILSTM"], "Only support CNN, LSTM or BILSTM."
    if args.model == "CNN":
            model = CNNModel(vocab_size, args.emb_size, args.max_seq_length, 
                    weights=vocab_weights, fix_emb_weight=args.fix_emb, num_classes=3)
    elif args.model == "LSTM":
            model = LSTMModel(vocab_size, args.emb_size, args.max_seq_length, 
                    weights=vocab_weights, fix_emb_weight=args.fix_emb, num_classes=3)
    else:
            model = BILSTMModel(vocab_size, args.emb_size, args.max_seq_length, 
                    weights=vocab_weights, fix_emb_weight=args.fix_emb, num_classes=3)
            

    model = model.to(device)

    # Train
    print("Training %s ..." % args.model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    #label_weight = WeightClassCSV(args.dataset + "/train.csv").get_weights(['0', '1'])
    #loss_func = nn.CrossEntropyLoss(weight = torch.tensor(label_weight).to(device))

    model.train()
    for epoch in trange(args.num_epoch, desc="Epoch"):
            total_loss = 0
            for idx, batch in enumerate(tqdm(train_iter, desc="Iteration")):
                    inputs, labels = batch.sent, batch.label
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)

                    loss = loss_func(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.data.item()

            print("\tEpoch %d, total loss: %f" % (epoch, total_loss))

    train_finish_time = time.time()
    train_overall_time = train_finish_time - start_time

    # Evaluate
    print("Evaluating ...")
    model.eval()
    predicts = []
    golds = []

    predicted_proba = []
    with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_iter, desc="Iteration")):
                    inputs, labels = batch.sent, batch.label
                    inputs = inputs.to(device)

                    logits = model(inputs)
                    predicted_proba += list(logits.data.cpu().numpy())

                    predict = torch.argmax(logits, dim=1).data.cpu().numpy()
                    predicts += list(predict)
                    golds += list(labels.data.cpu().numpy())

    precision, recall, f1 = F1(predicts, golds)
    print("Precision: %f, Recall: %f, F1: %f" % (precision, recall, f1))

    train_time = train_overall_time
    test_time = time.time() - train_finish_time

    (precision, recall, fscore, support) = metrics.precision_recall_fscore_support(golds, predicts)
    log_row = []
    log_row.append(args.dataset)
    log_row.append(fscore[0])
    log_row.append(fscore[1])
    log_row.append(fscore[2])


    csv_handler.append_row(args.log_file, log_row)
