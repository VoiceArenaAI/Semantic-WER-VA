from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

from .alignment import Segment, segments
from .backends import DEFAULT_MODEL, Backend, GeminiSync
from .cache import DEFAULT_PATH, VerdictCache
from .normalize import normalize
from .prompting import (available_languages, deletion_prompt, insertion_prompt, mark_deletion, mark_insertion,
                        substitution_prompt)

SUBSTITUTION, INSERTION, DELETION = "substitution", "insertion", "deletion"


@dataclass
class Counts:
    errors: int = 0
    ref_units: int = 0

    @property
    def rate(self) -> float:
        return self.errors / self.ref_units if self.ref_units else 0.0

    def add(self, other: "Counts") -> None:
        self.errors += other.errors
        self.ref_units += other.ref_units


def _word_counts(ref: str, hyp: str) -> Counts:
    r, h = ref.split(), hyp.split()
    if not r:
        return Counts(len(h), 0)
    if not h:
        return Counts(len(r), len(r))
    o = jiwer.process_words(ref, hyp)
    return Counts(o.substitutions + o.deletions + o.insertions, len(r))


def _char_counts(ref: str, hyp: str) -> Counts:
    if not ref:
        return Counts(len(hyp), 0)
    if not hyp:
        return Counts(len(ref), len(ref))
    o = jiwer.process_characters(ref, hyp)
    return Counts(o.substitutions + o.deletions + o.insertions, len(ref))


def _request(seg: Segment, ref_words: list[str]) -> tuple[str, str, str] | None:
    if seg.kind == "substitution":
        return SUBSTITUTION, " ".join(seg.ref), " ".join(seg.hyp)
    if seg.kind == "insertion":
        return INSERTION, mark_insertion(ref_words, seg.ref_start, list(seg.hyp)), ""
    if seg.kind == "deletion":
        return DELETION, mark_deletion(ref_words, seg.ref_start, seg.ref_start + len(seg.ref)), ""
    return None


@dataclass
class Utterance:
    reference: str
    hypothesis: str
    corrected: str
    segments: list[Segment]
    verdicts: list[bool | None]
    wer: float
    semantic_wer: float


@dataclass
class Result:
    language: str
    wer: float
    semantic_wer: float
    cer: float
    semantic_cer: float
    reference_words: int
    unique_substitutions: int
    substitutions_equal: int
    unique_insertions_deletions: int
    insertions_deletions_disfluent: int
    from_cache: int
    newly_judged: int
    unanswered: int
    utterances: list[Utterance] = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "utterances"}


class SemanticWER:

    def __init__(self, language: str, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 cache_path=DEFAULT_PATH, workers: int = 16, backend: Backend | None = None):
        if language not in available_languages():
            raise ValueError(f"unsupported language {language!r}; available: {available_languages()}")
        self.language = language
        self.backend = backend or GeminiSync(api_key, model, workers=workers)
        self.cache = VerdictCache(cache_path)

    def _key(self, req: tuple[str, str, str]) -> tuple:
        task, a, b = req
        return (task, self.language, self.backend.model, a, b)

    def _prompt(self, req: tuple[str, str, str]) -> str:
        task, a, b = req
        if task == SUBSTITUTION:
            return substitution_prompt(self.language, a, b)
        return (insertion_prompt if task == INSERTION else deletion_prompt)(self.language, a)

    def judge(self, requests: list[tuple[str, str, str]]) -> tuple[dict[tuple, bool | None], int, int]:
        unique = list(dict.fromkeys(requests))
        cached = self.cache.get_many([self._key(r) for r in unique])
        verdicts = {r: cached[self._key(r)] for r in unique if self._key(r) in cached}
        todo = [r for r in unique if r not in verdicts]
        if todo:
            fresh = dict(zip(todo, self.backend.judge([self._prompt(r) for r in todo])))
            verdicts.update(fresh)
            self.cache.put_many({self._key(r): v for r, v in fresh.items() if v is not None})
        return verdicts, len(unique) - len(todo), len(todo)

    def score(self, references: list[str], hypotheses: list[str]) -> Result:
        if len(references) != len(hypotheses):
            raise ValueError("references and hypotheses must have the same length")
        refs = [normalize(r) for r in references]
        hyps = [normalize(h) for h in hypotheses]
        segs = [segments(r, h) for r, h in zip(refs, hyps)]
        reqs = [[_request(s, r.split()) for s in ss] for r, ss in zip(refs, segs)]
        verdicts, n_cached, n_new = self.judge([q for rq in reqs for q in rq if q])

        utts, w, sw, c, sc = [], Counts(), Counts(), Counts(), Counts()
        for ref, hyp, ss, rq in zip(refs, hyps, segs, reqs):
            words, vs = [], []
            for s, q in zip(ss, rq):
                if q is None:
                    words.extend(s.ref)
                    continue
                v = verdicts.get(q)
                vs.append(v)
                if s.kind == "substitution":
                    words.extend(s.ref if v else s.hyp)
                elif s.kind == "deletion":
                    words.extend(s.ref if v else ())
                else:
                    words.extend(() if v else s.hyp)
            corrected = " ".join(words)
            wc, swc = _word_counts(ref, hyp), _word_counts(ref, corrected)
            w.add(wc), sw.add(swc), c.add(_char_counts(ref, hyp)), sc.add(_char_counts(ref, corrected))
            utts.append(Utterance(ref, hyp, corrected, ss, vs, wc.rate, swc.rate))

        unique = set(q for rq in reqs for q in rq if q)
        subs = [q for q in unique if q[0] == SUBSTITUTION]
        ids = [q for q in unique if q[0] in (INSERTION, DELETION)]
        return Result(
            language=self.language, wer=w.rate, semantic_wer=sw.rate, cer=c.rate, semantic_cer=sc.rate,
            reference_words=w.ref_units,
            unique_substitutions=len(subs), substitutions_equal=sum(1 for q in subs if verdicts.get(q)),
            unique_insertions_deletions=len(ids), insertions_deletions_disfluent=sum(1 for q in ids if verdicts.get(q)),
            from_cache=n_cached, newly_judged=n_new, unanswered=sum(1 for q in unique if verdicts.get(q) is None),
            utterances=utts)
