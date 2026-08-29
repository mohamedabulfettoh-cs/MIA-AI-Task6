"""
Captioning model. Pretrained CNN spatial features go through a linear
projection, then Bahdanau attention picks out which region matters at
each decoding step, then a GRU/LSTM writes the caption word by word.

Basically the "Show, Attend and Tell" setup. Kept the CNN output as a
grid (e.g. 8x8 for InceptionV3) instead of pooling it down to one
vector, otherwise attention has nothing to actually attend over.
"""
import torch
import torch.nn as nn

from src.config import CONFIG


class EncoderProjection(nn.Module):
    """Projects each spatial CNN feature vector into the decoder's embedding space."""

    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)  # LayerNorm so batch size 1 doesn't blow up at inference

    def forward(self, features):  # (B, N, feature_dim)
        return self.norm(self.proj(features))  # (B, N, embed_dim)


class BahdanauAttention(nn.Module):
    """Additive attention over the N spatial positions."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attn_dim)
        self.W_dec = nn.Linear(decoder_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, N, encoder_dim), decoder_hidden: (B, decoder_dim)
        score = self.V(torch.tanh(
            self.W_enc(encoder_out) + self.W_dec(decoder_hidden).unsqueeze(1)
        ))  # (B, N, 1)
        weights = torch.softmax(score, dim=1)
        context = (weights * encoder_out).sum(dim=1)
        return context, weights.squeeze(-1)


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = None,
        decoder_units: int = None,
        attn_dim: int = None,
        rnn_type: str = None,
        dropout: float = None,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()
        embed_dim = embed_dim or CONFIG.embedding_dim
        decoder_units = decoder_units or CONFIG.decoder_units
        attn_dim = attn_dim or CONFIG.attention_dim
        rnn_type = (rnn_type or CONFIG.decoder_type).lower()
        dropout = dropout if dropout is not None else CONFIG.dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.attention = BahdanauAttention(embed_dim, decoder_units, attn_dim)
        self.dropout = nn.Dropout(dropout)

        rnn_input_dim = embed_dim + embed_dim  # word embedding + attended context
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.rnn_cell = nn.LSTMCell(rnn_input_dim, decoder_units)
        else:
            self.rnn_cell = nn.GRUCell(rnn_input_dim, decoder_units)

        self.init_h = nn.Linear(embed_dim, decoder_units)
        self.init_c = nn.Linear(embed_dim, decoder_units)
        self.fc_out = nn.Linear(decoder_units, vocab_size)

    def init_hidden_state(self, encoder_out):  # (B, N, embed_dim)
        mean_enc = encoder_out.mean(dim=1)
        h = self.init_h(mean_enc)
        c = self.init_c(mean_enc)
        return h, c

    def forward(self, encoder_out, captions):
        """Teacher-forced forward pass. captions already have <start>...<end><pad>.
        Returns logits (B, T-1, vocab_size)."""
        B, T = captions.shape
        h, c = self.init_hidden_state(encoder_out)
        enc_memory = encoder_out

        embeddings = self.dropout(self.embedding(captions))  # (B, T, embed_dim)
        outputs = []
        for t in range(T - 1):
            context, _ = self.attention(enc_memory, h)
            rnn_in = torch.cat([embeddings[:, t, :], context], dim=1)
            if self.rnn_type == "lstm":
                h, c = self.rnn_cell(rnn_in, (h, c))
            else:
                h = self.rnn_cell(rnn_in, h)
            outputs.append(self.fc_out(self.dropout(h)))
        return torch.stack(outputs, dim=1)  # (B, T-1, vocab_size)

    @torch.no_grad()
    def generate(self, encoder_out, tokenizer, max_len=None, beam_size=1):
        """Greedy or beam search decoding, batch size 1 only."""
        max_len = max_len or CONFIG.max_caption_len
        device = encoder_out.device
        start_id = tokenizer.word2idx[CONFIG.start_token]
        end_id = tokenizer.word2idx[CONFIG.end_token]

        if beam_size <= 1:
            h, c = self.init_hidden_state(encoder_out)
            enc_memory = encoder_out
            token = torch.tensor([start_id], device=device)
            result = []
            for _ in range(max_len):
                emb = self.embedding(token)
                context, _ = self.attention(enc_memory, h)
                rnn_in = torch.cat([emb, context], dim=1)
                if self.rnn_type == "lstm":
                    h, c = self.rnn_cell(rnn_in, (h, c))
                else:
                    h = self.rnn_cell(rnn_in, h)
                logits = self.fc_out(h)
                token = logits.argmax(dim=-1)
                if token.item() == end_id:
                    break
                result.append(token.item())
            return tokenizer.decode(result, strip_special=True)

        # beam search
        sequences = [(0.0, [start_id], self.init_hidden_state(encoder_out))]
        enc_memory = encoder_out
        for _ in range(max_len):
            candidates = []
            for score, seq, (h, c) in sequences:
                if seq[-1] == end_id:
                    candidates.append((score, seq, (h, c)))
                    continue
                token = torch.tensor([seq[-1]], device=device)
                emb = self.embedding(token)
                context, _ = self.attention(enc_memory, h)
                rnn_in = torch.cat([emb, context], dim=1)
                if self.rnn_type == "lstm":
                    nh, nc = self.rnn_cell(rnn_in, (h, c))
                else:
                    nh = self.rnn_cell(rnn_in, h)
                    nc = c
                logprobs = torch.log_softmax(self.fc_out(nh), dim=-1).squeeze(0)
                topk = torch.topk(logprobs, beam_size)
                for lp, idx in zip(topk.values, topk.indices):
                    candidates.append((score + lp.item(), seq + [idx.item()], (nh, nc)))
            candidates.sort(key=lambda x: x[0], reverse=True)
            sequences = candidates[:beam_size]
            if all(s[1][-1] == end_id for s in sequences):
                break

        best_seq = sequences[0][1]
        return tokenizer.decode(best_seq, strip_special=True)


class CaptioningModel(nn.Module):
    """Wraps the encoder projection and decoder together."""

    def __init__(self, feature_dim: int, vocab_size: int, **kwargs):
        super().__init__()
        embed_dim = kwargs.get("embed_dim", CONFIG.embedding_dim)
        self.encoder = EncoderProjection(feature_dim, embed_dim)
        self.decoder = CaptionDecoder(vocab_size, **kwargs)

    def forward(self, features, captions):
        enc_out = self.encoder(features)
        return self.decoder(enc_out, captions)

    @torch.no_grad()
    def caption_image(self, features, tokenizer, beam_size=None):
        # features: (1, N, feature_dim)
        self.eval()
        enc_out = self.encoder(features)
        beam_size = beam_size or CONFIG.beam_size
        return self.decoder.generate(enc_out, tokenizer, beam_size=beam_size)
