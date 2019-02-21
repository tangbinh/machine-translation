import math
import torch
import torch.nn as nn

from seq2seq import utils


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, pad_idx=0, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, self.embed_dim, self.pad_idx)

    @staticmethod
    def get_embedding(num_embed, embed_dim, pad_idx=None):
        """Build sinusoidal embeddings."""
        half_dim = embed_dim // 2
        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, dtype=torch.float) * -embed)
        embed = torch.arange(num_embed, dtype=torch.float).unsqueeze(1) * embed.unsqueeze(0)
        embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1).view(num_embed, -1)
        if embed_dim % 2 == 1:
            embed = torch.cat([embed, torch.zeros(num_embed, 1)], dim=1)
        if pad_idx is not None:
            embed[pad_idx, :] = 0
        return embed.cuda()

    def forward(self, input, incremental_state=None, timestep=None):
        bsz, seq_len = input.size()
        max_pos = self.pad_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.pad_idx)

        if incremental_state is not None:
            # Positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            return self.weights[self.pad_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(input, self.pad_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
