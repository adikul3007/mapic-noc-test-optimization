import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple
from src.utils.tensor_utils import rearrange, gumbel_like, gumbel_with_maximum
from src.models.attention import Encoder, Decoder, Attention

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device, logit_clipping=True, decoding_type = 'sampling'):
        # print("using lstm")
        super().__init__()
        self.encoder: Encoder = Encoder(input_dim, hidden_dim, n_layers, p)
        self.decoder: Decoder = Decoder(input_dim, hidden_dim, n_layers, p)
        # self.decoder_tsv: Decoder_tsv = Decoder_tsv(input_dim, hidden_dim, n_layers, p)
        self.attn: Attention = Attention(hidden_dim, hidden_dim, logit_clipping)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))
        self.device = device
        self.decoding_type = decoding_type

    def preprocess(self, input, mask) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=1, dtype=torch.long)
        # seq_len is the length of the longest sequence in the batch
        seq_len = torch.max(lengths).item()
        # reshape input and mask to remove extra padding
        # first dimension of the input is seq_len
        input = input[:seq_len]
        # second dimension of the mask is seq_len
        mask = mask[:, :seq_len]
        return input, mask


    def forward(self, input: Tensor, num_io, mask, num_samples=1):

        # sample n_samples mapping solutions for each sequence in the batch
        # input, mask = self.preprocess(input, mask)
        # input shape: (seq_len, batch_size, input_dim)
        batch_size = input.size(1)
        seq_len = input.size(0)
        # print("inp shape",input.shape)
        # Tensor to store the predicted mapping
        # predicted_mappings shape: (batch * num_samples, seq_len)
        predicted_mappings = torch.zeros(batch_size*num_samples, seq_len, dtype=torch.int64).to(self.device)
        # print(mask.shape, 'ptr mask shape')
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        # first input should be a part of model learnable parameters
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        # mask to be used while calculating attention weights
        mask_decoding = mask.clone()
        log_probs_sum = torch.zeros(batch_size*num_samples, dtype=torch.float32).to(self.device)
        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            # logits shape: (batch * num_samples, seq_len)
            log_probs = F.log_softmax(logits, dim=1)
            # print(f"logits: {logits}")
            # print(f"log_probs shape: {log_probs}")
            # log_probs shape: (batch * num_samples, seq_len)
            if t == 0:
                # log_probs shape: (batch , seq_len)
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                    # selected_indices shape: (batch * num_samples,)
                        scores = log_probs + gumbel_like(log_probs)
                        # scores shape: (batch * num_samples, seq_len)
                        # print(f"scores shape: {scores.shape}")
                    elif self.decoding_type == 'greedy':
                        scores = log_probs
                    _, selected_indices = torch.topk(scores, min(num_samples, scores.size(-1)), dim=-1)
                    # selected_indices shape: (batch, min(num_samples, seq_len))
                    if num_samples > log_probs.size(1):
                        # pad second dimension with -1 so that the shape becomes (batch, num_samples)
                        selected_indices = F.pad(selected_indices, (0, num_samples - scores.size(1)), 'constant', -1)
                    selected_indices = selected_indices.view(-1)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples, replacement=True).long().view(-1)
                log_probs = log_probs.repeat_interleave(num_samples, dim=0)
                # print("lp1",log_probs.shape)
                # make log_probs -inf for the values which are -1 in selected_indices
                log_probs = log_probs.masked_fill((selected_indices == -1).unsqueeze(-1), float('-inf'))
                # print("lp2", log_probs.shape)
                if self.decoding_type == 'sampling-w/o-replacement':
                    scores = scores.repeat_interleave(num_samples, dim=0)
                    scores = scores.masked_fill((selected_indices == -1)\
                        .unsqueeze(-1), float('-inf'))
                mask_decoding = mask_decoding.repeat_interleave(num_samples, dim=0)
                # mask_decoding shape: (batch * num_samples, seq_len)
                hidden = hidden.repeat_interleave(num_samples, dim=1)
                cell = cell.repeat_interleave(num_samples, dim=1)
                encoder_outputs = encoder_outputs.repeat_interleave(num_samples, dim=1)
                # selected_indices = selected_indices % 5
            else:
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores, _ = gumbel_with_maximum(log_probs + log_probs_sum.unsqueeze(-1), g_log_probs)
                        # print(f"log_probs: {log_probs.shape}")
                        # print(f"log_probs_sum: {log_probs_sum.shape}")
                        # print(f"scores: {scores.shape}")
                    elif self.decoding_type == 'greedy':
                        scores = log_probs + log_probs_sum.unsqueeze(-1)
                    scores_per_batch = scores.view(batch_size, -1) # (64, 129)
                    # print(scores_per_batch.shape, 'scores shape')
                    top_scores, indices_buf = torch.topk(scores_per_batch, num_samples, dim=1)
                    # print("num_io: ",num_io)
                    # print("indices_buf before: ",indices_buf)
                    # print("indices_buf shape: ", indices_buf.shape)
                    # print("top_scores", top_scores.shape)
                    # indices_buf shape: (batch, min(num_samples, scores.size(1)))
                    # Reshape scores to (batch_size, num_samples * seq_len), but restrict to the first num_io positions
                    beams_buf = torch.div(indices_buf, num_io, rounding_mode='floor')
                    #print(beams_buf, 'beams')
                    indices_buf = indices_buf.fmod(num_io)
                    # indices_buf = indices_buf[:,-1]
                    # print("indices_buf: ", indices_buf)
                    selected_indices = indices_buf.view(-1)
                    predicted_mappings[:,:t] = rearrange(predicted_mappings[:,:t], beams_buf)
                    log_probs_sum = rearrange(log_probs_sum, beams_buf)
                    mask_decoding = rearrange(mask_decoding, beams_buf)
                    log_probs = rearrange(log_probs, beams_buf)
                    hidden = rearrange(hidden, beams_buf)
                    cell = rearrange(cell, beams_buf)
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores = rearrange(scores, beams_buf)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples).long().squeeze(1)
            predicted_mappings[:, t] = selected_indices
            # print(f"t={t}, sel={selected_indices}")
            gather_indices = selected_indices.unsqueeze(-1).clone()
            gather_indices[gather_indices == -1] = 0
            # print(f"log_probs: {log_probs}")
            # print(selected_indices, 'selected')
            # print(f"gather: {gather_indices}")
            # print(f"mask: {mask}")
            curr_log_probs = log_probs.gather(1, gather_indices).squeeze(-1) \
                * mask.repeat_interleave(num_samples, dim=0)[:, t]
            # print(f"curr_log_probs: {curr_log_probs}")
            log_probs_sum += curr_log_probs
            # print(f"log_probs_sum: {log_probs_sum}")
            if self.decoding_type == 'sampling-w/o-replacement':
                # gumbel perturbed log probabilities of partial sequences
                g_log_probs = scores.gather(1, gather_indices).squeeze(-1) \
                    * mask.repeat_interleave(num_samples, dim=0)[:, t]
            # decoder_input shape: (batch, input_dim)
            decoder_input = input.repeat_interleave(num_samples, dim=1)[gather_indices.squeeze(-1), torch.arange(batch_size * num_samples)]
            # update the mask_decoding to remove the pointed inputs
            mask_decoding.scatter_(1, gather_indices, 0)
        # assign -1 to the mappings corresponding to the padded values to denote that it is invalid
        predicted_mappings = predicted_mappings.masked_fill(mask.repeat_interleave(num_samples, dim=0) == 0, -1)
        predicted_mappings = predicted_mappings.view(batch_size, num_samples, seq_len).transpose(0, 1).reshape(-1,seq_len)
        log_probs_sum = log_probs_sum.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
        if self.decoding_type == 'sampling-w/o-replacement':
            g_log_probs = g_log_probs.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
            return predicted_mappings, log_probs_sum, g_log_probs
        else:
            return predicted_mappings, log_probs_sum