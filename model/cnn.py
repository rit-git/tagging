import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.nn import functional as F
import math

class CNNModel(torch.nn.Module):
	def __init__(self, vocab_size, emb_size, max_seq_length, kernel_size=5,
				 c_out=128, extra_kernels=[3, 4, 5], dropout=0.5, 
				 hidden_size=128, num_classes=2, weights=None, 
				 fix_emb_weight=False):
		super(CNNModel, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.max_seq_length = max_seq_length
		self.extra_kernels = extra_kernels 
		self.c_out = c_out
		self.kernel_size = kernel_size
		self.dropout = dropout
		self.hidden_size = hidden_size
		self.num_classes = num_classes

		# Define embedding layer
		self.emb_layer = nn.Embedding(vocab_size, emb_size)
		self.add_module("Embedding", self.emb_layer)
		if weights is None:
			xavier_uniform(self.emb_layer.weight.data)
		else:
			self.emb_layer.weight.data.copy_(weights)
		self.emb_layer.weight.requires_grad = not fix_emb_weight
		
		# Define CNN layer
		assert kernel_size % 2 == 1, "Invalid kernal size."
		padding = int((kernel_size-1)/2)
		self.conv1_layers = [torch.nn.Conv1d(emb_size, c_out, kernel_size, 
			padding=padding)]
		self.add_module("Conv1d_0", self.conv1_layers[0])

		for i in range(len(self.extra_kernels)):
			padding = int((self.extra_kernels[i]-1)/2)
			conv1 = torch.nn.Conv1d(c_out, c_out, self.extra_kernels[i], 
			padding=padding)
			self.add_module("Conv1d_%d" % (i+1), conv1)
			self.conv1_layers.append(conv1)

		self.pool = nn.MaxPool1d(kernel_size)
		self.dropout = nn.Dropout(dropout)
		
		# Update classification layer
		pool_size = math.floor((max_seq_length-kernel_size)/(kernel_size+0.0)+1)
		self.flat = nn.Linear(c_out*pool_size, hidden_size)
		self.linear = nn.Linear(hidden_size, num_classes)

	def forward(self, inputs):
		"""
		Parameters:
		Input (max_seq_length * batch_size tensor)
		"""
		# Update embedding
		inputs = inputs.transpose(1, 0)
		emb = self.emb_layer(inputs)
		emb = emb.transpose(1,2)

		# Update conv1 layers
		conv = emb
		for i in range(len(self.extra_kernels)):
			conv = self.conv1_layers[i](conv)
			conv = torch.nn.functional.relu(conv)
			conv = self.dropout(conv)

		# Update classifier
		conv = self.dropout(self.pool(conv))
		conv = conv.view(conv.size(0), -1)
		conv = self.dropout(self.flat(conv))
		logit = self.linear(conv)

		return logit
