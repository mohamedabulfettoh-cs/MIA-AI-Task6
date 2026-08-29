from src.text.tokenizer import Tokenizer


def test_fit_and_vocab_size():
    captions = ["a dog runs in the park", "a cat sleeps on the mat", "a dog barks"]
    tok = Tokenizer(min_freq=1).fit(captions)
    assert tok.vocab_size > len(tok._specials)
    assert "dog" in tok.word2idx


def test_encode_decode_roundtrip():
    captions = ["a man rides a bike down the street"]
    tok = Tokenizer(min_freq=1).fit(captions)
    ids = tok.encode("a man rides a bike", max_len=10)
    assert len(ids) == 10
    text = tok.decode(ids)
    assert "man" in text and "bike" in text


def test_unknown_word_maps_to_unk():
    tok = Tokenizer(min_freq=1).fit(["a small dog runs"])
    ids = tok.encode("a gigantic dog runs", max_len=8)
    unk_id = tok.word2idx["<unk>"]
    assert unk_id in ids
