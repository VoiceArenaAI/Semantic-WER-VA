from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="va-semantic-wer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--language", required=True, help="language code, e.g. en, pt, ro, vi")
    ap.add_argument("--ref", required=True, help="reference column")
    ap.add_argument("--hyp", required=True, nargs="+", help="hypothesis column(s)")
    ap.add_argument("--model", default=None, help="override the judge model id")
    ap.add_argument("--api-key", default=None, help="else GEMINI_API_KEY")
    ap.add_argument("--cache", default=None, help="cache file (default ~/.cache/voice_arena_semantic_wer/verdicts.jsonl)")
    ap.add_argument("--per-utterance", default=None, help="write per-utterance results to this CSV")
    args = ap.parse_args(argv)

    import pandas as pd

    from .cache import DEFAULT_PATH
    from .scorer import SemanticWER

    df = pd.read_csv(args.csv, keep_default_na=False)
    kw = {"model": args.model} if args.model else {}
    scorer = SemanticWER(args.language, api_key=args.api_key, cache_path=args.cache or DEFAULT_PATH, **kw)
    rows, per = [], []
    for col in args.hyp:
        res = scorer.score(df[args.ref].astype(str).tolist(), df[col].astype(str).tolist())
        rows.append({"model": col, **{k: (round(v * 100, 2) if k in ("wer", "semantic_wer", "cer", "semantic_cer") else v)
                                       for k, v in res.summary().items()}})
        if args.per_utterance:
            per += [{"row": i, "model": col, "reference": u.reference, "hypothesis": u.hypothesis,
                     "corrected": u.corrected, "wer": u.wer, "semantic_wer": u.semantic_wer} for i, u in enumerate(res.utterances)]
    print(pd.DataFrame(rows).to_string(index=False))
    if args.per_utterance:
        pd.DataFrame(per).to_csv(args.per_utterance, index=False)
        print(json.dumps({"per_utterance": args.per_utterance}))


if __name__ == "__main__":
    main()
