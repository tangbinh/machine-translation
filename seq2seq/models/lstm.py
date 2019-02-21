import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention', help='decoder attention')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        base_architecture(args)
        def build_embedding(dictionary, embed_dim, path=None):
            embedding = nn.Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad_idx)
            if path is not None:
                utils.load_embedding(path, dictionary, embedding)
            return embedding
        encoder_embedding = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        decoder_embedding = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)

        encoder = LSTMEncoder(
            dictionary=src_dict,
            embedding=encoder_embedding,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_num_layers,
            bidirectional=args.encoder_bidirectional,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )

        decoder = LSTMDecoder(
            dictionary=tgt_dict,
            embedding=decoder_embedding,
            hidden_size=args.decoder_hidden_size,
            num_layers=args.decoder_num_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_attention=bool(eval(args.decoder_use_attention)),
        )
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    def __init__(
        self, dictionary, embedding, hidden_size=512, num_layers=1, bidirectional=True,
        dropout_in=0.1, dropout_out=0.1
    ):
        super().__init__(dictionary)
        self.embedding = embedding
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )

    def forward(self, src_tokens, src_lengths):
        # Embed tokens and apply dropout
        bsz, seqlen = src_tokens.size()
        x = self.embedding(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Pack embedded tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths)

        # Apply LSTM
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x)

        # Unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_dim]

        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hiddens = combine_directions(final_hiddens)
            final_cells = combine_directions(final_cells)

        src_mask = src_tokens.eq(self.dictionary.pad_idx).t()
        return {
            'src_out': (x, final_hiddens, final_cells),
            'src_mask': src_mask if src_mask.any() else None,
        }


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.output_proj = nn.Linear(input_dim + output_dim, output_dim, bias=False)

    def forward(self, input, src_out, src_mask):
        # input:    bsz x input_dim
        # src_out:  src_len x bsz x output_dim

        x = self.input_proj(input)
        attn_scores = (src_out * x.unsqueeze(dim=0)).sum(dim=2)

        # Don't attend over padding
        if src_mask is not None:
            attn_scores.masked_fill_(src_mask, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=0)

        # Weighted sum of source hiddens
        x = (attn_scores.unsqueeze(dim=2) * src_out).sum(dim=0)

        # Combine with input and apply final projection
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    def __init__(
        self, dictionary, embedding, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, pretrained_embedding=None, use_attention=True,
    ):
        super().__init__(dictionary)
        self.embedding = embedding
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None
        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=hidden_size + embedding.embedding_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            ) for layer in range(num_layers)
        ])

        self.final_proj = nn.Linear(hidden_size, len(dictionary))

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]

        src_out, src_hiddens, src_cells = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        srclen = src_out.size(0)

        # Embed tokens and apply dropout
        bsz, seqlen = tgt_inputs.size()
        x = self.embedding(tgt_inputs)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            rnn_hiddens, rnn_cells, input_feed = cached_state
        else:
            # Initialize RNN cells with those from encoder
            rnn_hiddens = [src_hiddens[i] for i in range(len(self.layers))]
            rnn_cells = [src_cells[i] for i in range(len(self.layers))]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        rnn_outputs = []

        for j in range(seqlen):
            # Concatenate token embedding with output from previous time step
            input = torch.cat([x[j, :, :], input_feed], dim=1)

            for i, rnn in enumerate(self.layers):
                # Apply recurrent cell
                rnn_hiddens[i], rnn_cells[i] = rnn(input, (rnn_hiddens[i], rnn_cells[i]))

                # Hidden state becomes the input to the next layer
                input = F.dropout(rnn_hiddens[i], p=self.dropout_out, training=self.training)

            # Prepare input feed for next time step
            if self.attention is None:
                input_feed = rnn_hiddens[-1]
            else:
                input_feed, attn_scores[:, j, :] = self.attention(rnn_hiddens[-1], src_out, src_mask)
            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (rnn_hiddens, rnn_cells, input_feed))

        # Collect outputs across time steps
        x = torch.cat(rnn_outputs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # srclen x seqlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # Final projection
        x = self.final_proj(x)
        return x, attn_scores


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.1)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.1)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.1)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.1)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
