from __future__ import annotations

import re
import unicodedata

import ftfy

SPOKEN_SYMBOLS = frozenset("%٪‰&@")
APOSTROPHE_VARIANTS = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})
_STRAY_APOSTROPHE = re.compile(r"(?<![^\W_])'(?![^\W_])")
_SEPARATORS = ".,:٫٬"
_NUM_SEP = re.compile(rf"(?<=\d)[{re.escape(_SEPARATORS)}](?=\d)")
# a dot between two letters is kept ("ule.com", "b.com", "u.s"); the judge decides whether it matches
_INNER_DOT = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")
_TAG = re.compile(r"\[[^\]]*(?:\]|$)|<\|[^>]*>")
_PLACEHOLDER = "\uE000"
_DOT_PLACEHOLDER = "\uE001"


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P") and ch not in SPOKEN_SYMBOLS and ch != "'"


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    text = ftfy.fix_text(str(text))
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHE_VARIANTS)
    text = _NUM_SEP.sub(lambda m: _PLACEHOLDER + str(_SEPARATORS.index(m.group())), text)
    text = _INNER_DOT.sub(_DOT_PLACEHOLDER, text)
    text = "".join(" " if _is_punct(ch) else ch for ch in text)
    text = _STRAY_APOSTROPHE.sub(" ", text)
    text = re.sub(_PLACEHOLDER + r"(\d)", lambda m: _SEPARATORS[int(m.group(1))], text)
    text = text.replace(_DOT_PLACEHOLDER, ".")
    return re.sub(r"\s+", " ", text).strip()
