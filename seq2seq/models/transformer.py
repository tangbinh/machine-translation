import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils

from seq2seq.modules import MultiheadAttention, SinusoidalPositionalEmbedding
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, help='dropout probability after ReLU in FFN')
        parser.add_argument('--max-source-positions', default=1024, type=int, help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, help='max number of tokens in the target sequence')

        parser.add_argument('--encoder-embed-path', type=str, help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, help='num encoder attention heads')

        parser.add_argument('--decoder-embed-path', type=str, help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, help='num decoder attention heads')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        base_architecture(args)

        def build_embedding(dictionary, embed_dim, path=None):
            embedding = nn.Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad_idx)
            nn.init.normal_(embedding.weight, mean=0, std=embed_dim ** -0.5)
            nn.init.constant_(embedding.weight[dictionary.pad_idx], 0)
            if path is not None:
                utils.load_embedding(path, dictionary, embedding)
            return embedding

        encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


class TransformerEncoder(Seq2SeqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.dropout = args.dropout
        self.embed_scale = math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, args.max_source_positions + self.padding_idx + 1)
        self.layers = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        # Embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Encoder layer
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


class TransformerDecoder(Seq2SeqDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.dropout = args.dropout
        self.embed_scale = math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, args.max_source_positions + self.padding_idx + 1)
        self.layers = nn.ModuleList([TransformerDecoderLayer(args) for i in range(args.decoder_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        # Embed positions
        positions = self.embed_positions(tgt_inputs, incremental_state=incremental_state)
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
            positions = positions[:, -1:]

        # Embed tokens
        x = self.embed_scale * self.embed_tokens(tgt_inputs)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # Decoder layers
        for layer in self.layers:
            x, attn = layer(
                x, encoder_out['encoder_out'], encoder_out['encoder_padding_mask'],
                incremental_state, self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None)
            inner_states.append(x)
        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout

        self.self_attn = MultiheadAttention(self.embed_dim, args.encoder_attention_heads, args.attention_dropout)
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.layer_norms[0](x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout

        self.self_attn = MultiheadAttention(self.embed_dim, args.decoder_attention_heads, args.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MultiheadAttention(self.embed_dim, args.decoder_attention_heads, args.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(
        self, x, encoder_out, encoder_padding_mask, incremental_state, prev_self_attn_state=None,
        prev_attn_state=None, self_attn_mask=None, self_attn_padding_mask=None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state, need_weights=False, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        attn = None
        residual = x
        x = self.encoder_attn_layer_norm(x)
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True, need_weights=(not self.training))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
