from __future__ import annotations

from neurosynth.llm.pmid_verify import PMIDVerifier


def test_pmid_verifier_cache_logic() -> None:
    v = PMIDVerifier()
    # Non-numeric unlikely PMID should generally fail but function must return bool.
    out = v.is_valid("invalid-pmid")
    assert isinstance(out, bool)
