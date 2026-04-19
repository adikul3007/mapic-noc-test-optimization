import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                p):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                        dropout=p).to(device)

    def forward(self, input, mask):
        # input shape: (seq_len, batch, input_dim)
        # mask shape: (batch, seq_len)
        # print(input.shape, 'encoder input shape')
        # print(mask.shape, 'encoder mask shape')
        lengths = mask.sum(dim=1)
        # lengths shape: (batch)
        packed_inputs = pack_padded_sequence(input, lengths.to('cpu'), enforce_sorted=False).to(device)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        # output shape: (seq_len, batch, hidden_dim)
        # hidden shape: (n_layers, batch, hidden_dim) cell shape: (n_layers, batch, hidden_dim)
        # if encoder is bidirectional
        # pass hidden and cell through a linear layer to match decoder hidden and cell dimension
        seq_len = input.shape[0]
        # reshape output to (seq_len, batch, hidden_dim)
        return output, (hidden, cell)
    
class Decoder(nn.Module):
    # module for single step of decoding process
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super(Decoder, self).__init__()
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, hidden, cell):
        # input shape: (batch, input_dim)
        # hidden shape: (n_layers, batch, hidden_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim*2)
        # output shape: (batch, input_dim)
        input = input.unsqueeze(0).to(device)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)
    
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, logit_clipping=True, clip_value=10):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, dec_dim),
            nn.Tanh(),
        ).to(device)
        self.v = nn.Linear(dec_dim, 1, bias=False).to(device)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

    def forward(self, decoder_output, encoder_outputs, num_io, mask):
        # decoder_output is the output of the decoder of single step
        # encoder_outputs is a list of all the encoder outputs
        # decoder_output shape: (batch, dec_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim)
        # print(decoder_output.shape, " decoder shape")
        # print(encoder_outputs.shape, " encoder shape")
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1).to(device)
        encoder_outputs = encoder_outputs.permute(1, 0, 2).to(device)
        # decoder_output shape: (batch, seq_len, dec_dim)
        # encoder_outputs shape: (batch, seq_len, enc_dim)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2)).to(device)
        # energy shape: (batch, seq_len, dec_dim)
        attention = self.v(energy).squeeze(2).to(device)
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
        # use mask to remove the attention weights for padded values
        # print(attention.shape, 'attention')
        # print(mask.shape, 'mask')
        attention = attention.masked_fill(mask == 0, float('-inf'))
        # attention shape: (batch, seq_len)
        attentions, _ = torch.topk(attention, k=num_io, dim=1)
        attentions = attentions.to(device)
        return attentions