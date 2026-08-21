from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig


def test_tokenizer_is_lossless_for_unicode_and_unseen_text() -> None:
    tokenizer = BytePhraseTokenizer.fit(
        ["cell membrane cell membrane"],
        TokenizerConfig(max_phrases=32, min_frequency=2),
    )
    text = "cell membrane — Δ unseen"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size > 256


def test_phrase_tokens_reduce_repeated_text() -> None:
    text = "pyramidal pyramidal pyramidal"
    tokenizer = BytePhraseTokenizer.fit([text])
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))

