from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

EXAMPLE_RULES = {"substitution": 7, "insertion": 4, "deletion": 4}
INSERT_OPEN, DELETE_OPEN, CLOSE = "⟨+", "⟨−", "⟩"
_FOLDER = {"substitution": "substitution", "insertion": "insdel", "deletion": "insdel"}
_CORE = {"substitution": "core.txt", "insertion": "core_insertion.txt", "deletion": "core_deletion.txt"}


def _read(task: str, name: str) -> str:
    return resources.files(f"va_semantic_wer.prompts.{_FOLDER[task]}").joinpath(name).read_text(encoding="utf-8")


def available_languages() -> list[str]:
    langs = None
    for folder in set(_FOLDER.values()):
        root = resources.files(f"va_semantic_wer.prompts.{folder}")
        found = {p.name.removesuffix(".json") for p in root.iterdir() if p.name.endswith(".json")}
        langs = found if langs is None else langs & found
    return sorted(langs)


def _verdict(e: dict) -> str:
    return "True" if e["equal"] else "False"


def _substitution_lines(examples: list[dict]) -> str:
    rows = [(f'"{e["reference"]}"', f'"{e["prediction"]}"', _verdict(e), e.get("note", "")) for e in examples]
    w_ref = max(len(r) for r, *_ in rows)
    w_pred = max(len(p) for _, p, *_ in rows)
    return "\n".join(f"   Reference: {r:<{w_ref}}  Prediction: {p:<{w_pred}}  -> {v}" + (f"  ({n})" if n else "")
                     for r, p, v, n in rows)


def _insdel_lines(examples: list[dict], marker: str) -> str:
    rows = [(f'"{e["sentence"].replace("⟨ ", marker + " ")}"', _verdict(e), e.get("note", "")) for e in examples]
    w = max(len(s) for s, *_ in rows)
    return "\n".join((f"   Reference: {s:<{w}}  -> {v:<5}" + (f"  ({n})" if n else "")).rstrip() for s, v, n in rows)


@lru_cache(maxsize=None)
def template(task: str, language: str) -> str:
    if language not in available_languages():
        raise ValueError(f"no prompt for language {language!r}; available: {available_languages()}")
    spec = json.loads(_read(task, f"{language}.json"))
    marker = INSERT_OPEN if task == "insertion" else DELETE_OPEN
    lines = _substitution_lines if task == "substitution" else (lambda ex: _insdel_lines(ex, marker))
    fields = {"language_name": spec["language_name"]}
    for i in range(1, EXAMPLE_RULES[task] + 1):
        fields[f"examples_{i}"] = lines(spec["examples"][str(i)])
        note = spec.get("notes", {}).get(str(i))
        fields[f"note_{i}"] = f"\n   {note}" if note else ""
    return _read(task, _CORE[task]).format(**fields, ref="{ref}", pred="{pred}", sentence="{sentence}")


def substitution_template(language: str) -> str:
    return template("substitution", language)


def insertion_template(language: str) -> str:
    return template("insertion", language)


def deletion_template(language: str) -> str:
    return template("deletion", language)


def substitution_prompt(language: str, ref: str, pred: str) -> str:
    return substitution_template(language).replace("{ref}", ref).replace("{pred}", pred)


def mark_insertion(ref_words: list[str], position: int, inserted: list[str]) -> str:
    return " ".join(ref_words[:position] + [INSERT_OPEN, *inserted, CLOSE] + ref_words[position:])


def mark_deletion(ref_words: list[str], start: int, end: int) -> str:
    return " ".join(ref_words[:start] + [DELETE_OPEN, *ref_words[start:end], CLOSE] + ref_words[end:])


def insertion_prompt(language: str, marked_sentence: str) -> str:
    return insertion_template(language).replace("{sentence}", marked_sentence)


def deletion_prompt(language: str, marked_sentence: str) -> str:
    return deletion_template(language).replace("{sentence}", marked_sentence)
