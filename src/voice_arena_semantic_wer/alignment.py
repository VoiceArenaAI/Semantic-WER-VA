from __future__ import annotations

from dataclasses import dataclass

import jiwer


@dataclass(frozen=True)
class Segment:
    kind: str
    ref: tuple[str, ...]
    hyp: tuple[str, ...]
    ref_start: int


def _chunks(ref_words: list[str], hyp_words: list[str]):
    if not ref_words and not hyp_words:
        return []
    if not ref_words:
        return [("insert", 0, 0, 0, len(hyp_words))]
    if not hyp_words:
        return [("delete", 0, len(ref_words), 0, 0)]
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    return [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx) for c in out.alignments[0]]


def segments(ref: str, hyp: str) -> list[Segment]:
    ref_words, hyp_words = ref.split(), hyp.split()
    result: list[Segment] = []
    pending = None
    for kind, rs, re_, hs, he in _chunks(ref_words, hyp_words):
        if kind == "equal":
            if pending:
                result.append(_close(pending, ref_words, hyp_words))
                pending = None
            result.append(Segment("equal", tuple(ref_words[rs:re_]), tuple(hyp_words[hs:he]), rs))
        elif pending is None:
            pending = [rs, re_, hs, he]
        else:
            pending = [min(pending[0], rs), max(pending[1], re_), min(pending[2], hs), max(pending[3], he)]
    if pending:
        result.append(_close(pending, ref_words, hyp_words))
    return result


def _close(p, ref_words, hyp_words) -> Segment:
    r, h = tuple(ref_words[p[0]:p[1]]), tuple(hyp_words[p[2]:p[3]])
    kind = "substitution" if r and h else ("deletion" if r else "insertion")
    return Segment(kind, r, h, p[0])
