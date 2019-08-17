import torch
import torch.nn as nn
from torch.autograd import Variable

# nn.Module Documentation: https://github.com/torch/nn/blob/master/doc/module.md
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        """
            input_size = # of unique chars in the dataset
            hidden_size = number of neurons
            output_size = # of unique chars in the output. For generation, it equals to input_size.
                          For classification, it can be other things.
        
            model = which cell do you use. "gru" or "lstm"
            
            n_layer = number of hidden layers for the cell
        """
        
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size) # sparse word embedding
        
        self.model = model
        if model.lower() == "lstm":
            self.cell = nn.LSTM(hidden_size, hidden_size, n_layers)
        else: # model is GRU
            self.cell = nn.GRU(hidden_size, hidden_size, n_layers) # GRU unit

        self.linear = nn.Linear(hidden_size, output_size) # Fully connected layer

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.embedding(input)
        output, hidden = self.cell(encoded.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden
    

    def init_hidden(self, batch_size):
        if self.model.lower() == "lstm":
            return (
                    Variable( torch.zeros(self.n_layers, batch_size, self.hidden_size) ), 
                    Variable( torch.zeros(self.n_layers, batch_size, self.hidden_size) ) 
                    )
        else:
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        
