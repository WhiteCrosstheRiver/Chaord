"""A14 -- RED: the documentation gate is a substring test.

check_a14 decides 'reference.md covers key K' by regexing for the bare word
K anywhere in the file.  A reference consisting of unrelated prose
('We cell things carefully. We seed things carefully. ...') passes 35/35
coverage with zero real entries.  Real drift scenario: an entry is deleted
or renamed but the word survives in prose -> the gate stays green.

Measured (2026-09-30): prose-only reference -> A14 PASS, 'covers 35/35'.
The 9 keys with '(no example in spec/examples/)' additionally show the
example half of the criterion is also unenforced (it is reported, not
gated).

Desired: coverage must require an entry-shaped pattern (a heading, a table
row, or a `` `key` `` code span), not any word occurrence.
"""
from tools import acceptance as acc


def test_a14_prose_only_reference_is_rejected():
    keys = acc.check_a14()["details"]["keys"]
    prose = " ".join(f"We {k} things carefully." for k in keys)
    try:
        r = acc.check_a14(reference_text=prose)
        assert not r["passed"], (
            f"A14 RED: a reference made of unrelated sentences passes "
            f"coverage for all {len(keys)} keys ({r['evidence'][:80]}); the "
            "gate cannot distinguish a reference entry from an incidental "
            "word"
        )
    finally:
        # check_a14 regenerates docs/reference_generated.md as a side effect;
        # leave it regenerated against the real reference.md
        acc.check_a14()
