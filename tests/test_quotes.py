from extract import normalize_quote, quote_in_text
from validate import kim_alignment, quote_row

TEXT = "[WEAKNESSES]\nThe paper’s  evaluation uses only CIFAR-10 –\nno ImageNet results.\n“Why no error bars?”"


def test_quote_matches_despite_case_whitespace_and_typography():
    assert quote_in_text("the paper's evaluation uses only CIFAR-10 - no ImageNet results.", TEXT)
    assert quote_in_text('"Why no error bars?"', TEXT)


def test_paraphrase_is_not_a_quote():
    assert not quote_in_text("The evaluation covers only CIFAR-10", TEXT)


def test_normalize_collapses_whitespace_and_drops_backslashes():
    assert normalize_quote("  A\tB\n C ") == "a b c"
    assert normalize_quote('\\"sparsely modulated\\"') == '"sparsely modulated"'


REVIEW = (
    "[SUMMARY] The paper proposes a new method for training sparse networks at scale.\n"
    "[STRENGTHS] The writing is clear and the figures are easy to read throughout.\n"
    "[WEAKNESSES] There is no comparison with the RigL baseline on ImageNet at all.\n"
    "The proof of Theorem 2 silently assumes bounded gradients everywhere.\n"
    "[QUESTIONS] Why are there no error bars in Table 3 of the main paper?"
)


def test_quote_row_counts_found_duplicates_and_uncovered_sentences():
    claims = [
        {"quote": "no comparison with the RigL baseline on ImageNet"},
        {"quote": "comparison with the RigL baseline on ImageNet at all"},
        {"quote": "The proof of Theorem 2 silently assumes bounded gradients"},
        {"quote": "step 3 of the argument does not follow"},
    ]
    assert quote_row(REVIEW, claims) == dict(claims=4, found=3, duplicated=2, sentences=3, uncovered=1)


def test_quote_row_ignores_summary_and_strengths():
    assert quote_row(REVIEW, []) == dict(claims=0, found=0, duplicated=0, sentences=3, uncovered=3)


def test_kim_alignment():
    items = [
        "There is no comparison with the RigL baseline on ImageNet at all.",
        "The proof of Theorem 2 silently assumes bounded gradients everywhere.",
        "Why are there no error bars in Table 3 of the main paper?",
    ]
    claims = [
        {"quote": "no comparison with the RigL baseline on ImageNet"},
        {"quote": "proof of Theorem 2 silently assumes bounded gradients everywhere. Why are there no error bars"},
        {"quote": "the RigL baseline on ImageNet at all"},
        {"quote": "something the reviewer never wrote in this review text"},
    ]
    assert kim_alignment(items, claims) == dict(
        items=3, claims=4, items_hit=3, items_split=1, claims_unmatched=1, claims_spanning=1
    )


def test_short_quotes_match_by_their_full_length():
    items = ["Figure 2: the font is too small to read.", "P8, last para, L1: mases -> masses"]
    claims = [{"quote": "font is too small"}, {"quote": "mases -> masses"}, {"quote": "font"}]
    assert kim_alignment(items, claims)["items_hit"] == 2
    assert kim_alignment(items, claims)["claims_unmatched"] == 0
