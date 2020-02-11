import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.nn import functional as F

class BILSTMModel(torch.nn.Module):
	def __init__(self, vocab_size, emb_size, max_seq_length, dropout=0.5, 
				 hidden_size=256, num_layers=2, num_classes=2, weights=None, 
				 fix_emb_weight=False, bidirectional=True):
		super(BILSTMModel, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.max_seq_length = max_seq_length
		self.dropout = dropout
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_classes = num_classes
		self.bidirectional = bidirectional
		self.mul = 2 if self.bidirectional else 1

		# Define embedding layer
		self.emb_layer = nn.Embedding(vocab_size, emb_size)
		self.add_module("Embedding", self.emb_layer)
		if weights is None:
			xavier_uniform(self.emb_layer.weight.data)
		else:
			self.emb_layer.weight.data.copy_(weights)
		self.emb_layer.weight.requires_grad = not fix_emb_weight

		# Define LSTM layer
		self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, 
			bidirectional=bidirectional)
		self.flat = nn.Linear(hidden_size*self.mul, 1)
		self.linear = nn.Linear(max_seq_length, num_classes)

		self.dropout = nn.Dropout(dropout)

	def attention_net(self, lstm_output, final_state):
		# Disabled
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), 
									  soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state

	def forward(self, inputs):
		"""
		Parameters:
		Input (max_seq_length * batch_size tensor)
		"""
		# Update embedding
		inputs = inputs.transpose(1, 0)
		emb = self.emb_layer(inputs)

		# Update hidden layers
		h_0, c_0 = self.initial_vectors(emb.size(0), emb.is_cuda)
		output, hidden = self.lstm(emb.transpose(1,0), (h_0, c_0))
		output = self.flat(output.transpose(1,0)).squeeze(2)
		#output=self.attention_net(output, hidden[0])

		# Update classifier
		logits = self.linear(output)

		return logits

	def initial_vectors(self, batch_size, is_cuda):
		h_0 = Variable(torch.zeros(self.num_layers*self.mul, batch_size, 
			self.hidden_size))
		c_0 = Variable(torch.zeros(self.num_layers*self.mul, batch_size, 
			self.hidden_size))
		if is_cuda:
			return h_0.to("cuda"), c_0.to("cuda")
		else:
			return h_0, c_0
			
