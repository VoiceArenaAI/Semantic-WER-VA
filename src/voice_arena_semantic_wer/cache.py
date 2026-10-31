from __future__ import annotations

import json
import os
import threading
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("VA_SEMANTIC_WER_CACHE", Path.home() / ".cache" / "voice_arena_semantic_wer" / "verdicts.jsonl"))
FIELDS = ("task", "language", "model", "ref", "hyp")


class VerdictCache:
    def __init__(self, path: str | Path | None = DEFAULT_PATH):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[tuple, bool] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self._data[tuple(row[f] for f in FIELDS)] = bool(row["equal"])

    def get_many(self, keys: list[tuple]) -> dict[tuple, bool]:
        return {k: self._data[k] for k in keys if k in self._data}

    def put_many(self, items: dict[tuple, bool]) -> None:
        if not items:
            return
        with self._lock:
            self._data.update(items)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    for k, v in items.items():
                        fh.write(json.dumps({**dict(zip(FIELDS, k)), "equal": v}, ensure_ascii=False) + "\n")
