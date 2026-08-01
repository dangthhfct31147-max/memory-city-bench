"""Unit tests for the Phase 7 deterministic evidence compressor.

The compressor is dependency-free and deterministic, so these tests run without
any extra. They verify:

* high-signal sentences (query overlap, entities, numbers, negations) are kept
  and filler is dropped;
* citation ids and passage ordering survive compression;
* a context-poor passage is never emptied (recall safety);
* token accounting and the aggregate compression ratio are correct.
"""

from __future__ import annotations

from memcity.readers.compression import EvidenceCompressor


def test_keeps_query_overlap_and_drops_filler():
    comp = EvidenceCompressor(max_sentences_per_passage=1)
    text = (
        "The weather was pleasant that morning. "
        "Alice migrated the database to PostgreSQL. "
        "We then went for lunch."
    )
    out, n_kept, n_total = comp.compress_passage(text, "which database did Alice migrate to")
    assert n_total == 3
    assert n_kept == 1
    assert "PostgreSQL" in out
    assert "weather" not in out


def test_keeps_numbers_dates_and_negations():
    comp = EvidenceCompressor(max_sentences_per_passage=3, min_query_overlap=99)
    text = (
        "Nothing much happened here. "
        "The budget was 4200 dollars in 2023. "
        "The client did not approve the plan. "
        "A totally unrelated filler sentence about clouds."
    )
    # Query overlap is effectively disabled (min_query_overlap=99); retention must
    # come from the numeric and negation signals alone.
    out, n_kept, _ = comp.compress_passage(text, "zzz")
    assert "4200" in out          # numeric signal
    assert "did not approve" in out  # negation signal
    assert "clouds" not in out


def test_context_poor_passage_never_emptied():
    comp = EvidenceCompressor(max_sentences_per_passage=2)
    # No query overlap, no entity, no number, no negation, multiple sentences.
    text = "it was a thing. then another thing. and more."
    out, n_kept, n_total = comp.compress_passage(text, "quantum chromodynamics")
    assert out  # never empty
    assert n_kept >= 1
    assert n_total == 3


def test_single_sentence_passage_returned_verbatim():
    comp = EvidenceCompressor()
    out, n_kept, n_total = comp.compress_passage("Just one sentence here", "anything")
    assert out == "Just one sentence here"
    assert n_kept == n_total == 1


def test_compress_evidence_preserves_ids_and_order():
    comp = EvidenceCompressor(max_sentences_per_passage=1)
    evidence = [
        {"id": "ep-1", "text": "Filler one. Alice chose PostgreSQL. Filler two.",
         "source_episode_ids": ["ep-1"]},
        {"id": "ep-2", "text": "Bob booked the 9:30 flight. Small talk here.",
         "source_episode_ids": ["ep-2"]},
    ]
    out, stats = comp.compress_evidence(evidence, "database flight")
    # Ids and order preserved; extra fields carried through untouched.
    assert [e["id"] for e in out] == ["ep-1", "ep-2"]
    assert out[0]["source_episode_ids"] == ["ep-1"]
    # Compression removed tokens overall.
    assert stats.tokens_after < stats.tokens_before
    assert 0.0 < stats.compression_ratio <= 1.0
    assert stats.passages == 2


def test_compress_evidence_does_not_mutate_input():
    comp = EvidenceCompressor(max_sentences_per_passage=1)
    evidence = [{"id": "ep-1", "text": "One. Two. Three three three."}]
    original_text = evidence[0]["text"]
    comp.compress_evidence(evidence, "two")
    assert evidence[0]["text"] == original_text  # input dict untouched


def test_compression_ratio_zero_when_nothing_to_remove():
    comp = EvidenceCompressor()
    evidence = [{"id": "ep-1", "text": "Single sentence"}]
    _, stats = comp.compress_evidence(evidence, "single")
    assert stats.compression_ratio == 0.0


def test_disabling_signals_narrows_retention():
    # With entity/numeric/negation off and no query overlap, only the fallback
    # (leading sentence) survives.
    comp = EvidenceCompressor(
        max_sentences_per_passage=3,
        keep_numeric=False,
        keep_entities=False,
        keep_negations=False,
        min_query_overlap=99,
    )
    text = "Alice paid 500 dollars. She did not refund it. Extra sentence."
    out, n_kept, _ = comp.compress_passage(text, "zzz")
    assert n_kept == 1
    assert out == "Alice paid 500 dollars."  # fallback: first sentence
