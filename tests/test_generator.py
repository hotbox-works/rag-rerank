import pytest

import generator


@pytest.mark.parametrize("raw, expected", [
    ("Answer: Paris", "Paris"),
    ("answer:  The Beatles.", "The Beatles"),
    ("Let me think.\nAnswer: 1998\n", "1998"),
    ("Answer: Yes, both are American.", "yes"),
    ("Answer: No.", "no"),
    ("Answer: No Doubt", "No Doubt"),
    ("Answer: \"Blue Velvet\"", "Blue Velvet"),
    ("**Answer: Lisbon**", "Lisbon"),
    ("Chicago", "Chicago"),
    ("Answer: first\nAnswer: second", "second"),
    ("", ""),
])
def test_parse_answer(raw, expected):
    assert generator.parse_answer(raw) == expected


def test_answer_messages_context_and_closed_book():
    ctx = [{"title": "Oasis", "text": "formed 1991"}, {"title": "Blur", "text": "formed 1988"}]
    user = generator.answer_messages("Which formed first?", ctx)[-1]["content"]
    assert "Document 1 (Oasis): formed 1991" in user and "Document 2 (Blur)" in user
    assert user.index("Document 1") < user.index("Question:")  # context before question, as in Part 3
    closed = generator.answer_messages("Which formed first?", [])[-1]["content"]
    assert "Document" not in closed and "Question: Which formed first?" in closed


def test_summary_messages_and_clean():
    user = generator.summary_messages("Q?", {"title": "T", "text": "body"})[-1]["content"]
    assert user.startswith("Passage (T):\nbody") and "Question: Q?" in user
    assert generator.clean_summary("Not relevant. The passage is about cats.") == generator.NOT_RELEVANT
    assert generator.clean_summary(" Oasis formed in 1991. ") == "Oasis formed in 1991."


def test_generate_batch_with_fake(fake_llm):
    answers, raw = generator.generate_batch(fake_llm, [("q1", [{"title": "A", "text": "x"}]), ("q2", [])])
    assert answers == ["A", "unknown"] and len(raw) == 2
