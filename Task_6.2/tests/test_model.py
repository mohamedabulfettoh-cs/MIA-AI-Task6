import torch

from src.models.caption_model import CaptioningModel


def test_forward_pass_output_shape():
    batch_size, num_positions, feature_dim, vocab_size, seq_len = 4, 64, 2048, 50, 10
    model = CaptioningModel(feature_dim, vocab_size, embed_dim=32, decoder_units=64, attn_dim=32)

    features = torch.randn(batch_size, num_positions, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(features, captions)
    assert logits.shape == (batch_size, seq_len - 1, vocab_size)


def test_forward_pass_works_with_batch_size_one():
    num_positions, feature_dim, vocab_size, seq_len = 64, 2048, 50, 10
    model = CaptioningModel(feature_dim, vocab_size, embed_dim=32, decoder_units=64, attn_dim=32)
    features = torch.randn(1, num_positions, feature_dim)
    captions = torch.randint(0, vocab_size, (1, seq_len))
    logits = model(features, captions)
    assert logits.shape == (1, seq_len - 1, vocab_size)


def test_attention_weights_are_a_real_distribution_over_positions():
    from src.models.caption_model import BahdanauAttention

    attn = BahdanauAttention(encoder_dim=32, decoder_dim=64, attn_dim=16)
    encoder_out = torch.randn(2, 49, 32)   # (B, N=49, encoder_dim)
    decoder_hidden = torch.randn(2, 64)
    context, weights = attn(encoder_out, decoder_hidden)

    assert weights.shape == (2, 49)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)
    assert context.shape == (2, 32)


def test_generate_returns_string():
    from src.text.tokenizer import Tokenizer

    tok = Tokenizer(min_freq=1).fit(["a dog runs in the park"])
    model = CaptioningModel(2048, tok.vocab_size, embed_dim=32, decoder_units=64, attn_dim=32)
    features = torch.randn(1, 64, 2048)
    caption = model.caption_image(features, tok, beam_size=1)
    assert isinstance(caption, str)
