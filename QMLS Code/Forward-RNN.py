# load torch 
import torch  
import torch.nn as nn

# Define the MLMG class as a subclas of nn.Modul es
class MLMG(nn.Module):
    # Define the constructor with input_size, hidden_size, output_size as parameters
    def __init__(self, input_size, hidden_size, output_size):
        # Call the superclass constructor
        super(MLMG, self).__init__()
        # Define the hidden_size attribute
        self.hidden_size = hidden_size
        # Define the forward-RNN cell as an attribute, using nn.RNNCell with input_size, hidden_size, and tanh activation
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')
        # Define the linear layer as an attribute, using nn.Linear with hidden_size and output_size
        self.fc = nn.Linear(hidden_size, output_size)
        # Define the softmax layer as an attribute, using nn.LogSoftmax with dim=1
        self.softmax = nn.LogSoftmax(dim=1)

    # Define the forward method with x as the input parameter
    def forward(self, x):
        # Initialize the hidden state as a zero tensor with shape (x.size(1), self.hidden_size)
        h = torch.zeros(x.size(1), self.hidden_size)
        # Loop over the sequence length dimension of x (x.size(0))
        for i in range(x.size(0)):
            # Update the hidden state by passing x[i,:,:] and h to the rnn_cell
            h = self.rnn_cell(x[i,:,:], h)
        # Pass the final hidden state to the linear layer and store the result as out
        out = self.fc(h)
        # Pass out to the softmax layer and return result
        return self.softmax(out)
