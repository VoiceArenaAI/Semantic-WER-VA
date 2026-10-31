# Voice Arena Semantic WER

Official scorer for the Voice Arena Conversational ASR Leaderboard.

Standard WER counts every difference between a transcript and the reference as an
error, even when it is the same words written differently ("I'm" vs "I am",
"5" vs "five") or only a disfluency such as "uh". This scorer aligns the transcript
with the reference, asks an LLM judge (Gemini 3.6 Flash) whether each difference
is a real error, and reports semantic WER and CER alongside standard WER and CER.

## Install

```bash
pip install va-semantic-wer
```

## Use

```python
from va_semantic_wer import SemanticWER

scorer = SemanticWER("en")          # reads GEMINI_API_KEY
res = scorer.score(references, hypotheses)
print(res.wer, res.semantic_wer)
```

Languages: en, pt, ro, vi.
