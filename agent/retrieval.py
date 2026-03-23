#!/usr/bin/env python3
"""Local, dependency-free retrieval layer for STS2 agent context."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TEXT_EXTENSIONS = {".md", ".txt", ".json", ".jsonl"}


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class RetrievalHit:
    source: str
    title: str
    content: str
    score: float
    start_line: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "score": round(self.score, 4),
            "start_line": self.start_line,
        }


@dataclass
class _Chunk:
    source: str
    title: str
    content: str
    start_line: int
    term_freq: Dict[str, int]


class LocalRetriever:
    """Keyword-based retrieval over local docs and memory artifacts."""

    def __init__(self, chunks: List[_Chunk]) -> None:
        self.chunks = chunks
        self.doc_freq: Dict[str, int] = {}
        for chunk in chunks:
            for token in set(chunk.term_freq):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    @classmethod
    def from_paths(cls, paths: Iterable[str]) -> "LocalRetriever":
        chunks: List[_Chunk] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                        chunks.extend(cls._load_file(child))
            elif path.is_file():
                chunks.extend(cls._load_file(path))
        return cls(chunks)

    @staticmethod
    def _load_file(path: Path) -> List[_Chunk]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        lines = text.splitlines()
        chunks: List[_Chunk] = []
        current: List[str] = []
        current_title = path.name
        start_line = 1

        def flush() -> None:
            nonlocal current, start_line
            content = "\n".join(current).strip()
            if not content:
                current = []
                return
            tokens = _tokenize(content)
            if not tokens:
                current = []
                return
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            chunks.append(
                _Chunk(
                    source=str(path),
                    title=current_title,
                    content=content,
                    start_line=start_line,
                    term_freq=term_freq,
                )
            )
            current = []

        for index, line in enumerate(lines, start=1):
            if line.startswith("#"):
                flush()
                current_title = line.lstrip("# ").strip() or path.name
                start_line = index
            if not current:
                start_line = index
            current.append(line)
            if sum(len(item) for item in current) > 900:
                flush()
        flush()
        return chunks

    def search(self, query: str, limit: int = 3) -> List[RetrievalHit]:
        tokens = _tokenize(query)
        if not tokens or not self.chunks:
            return []

        results: List[RetrievalHit] = []
        doc_count = max(len(self.chunks), 1)
        for chunk in self.chunks:
            score = 0.0
            for token in tokens:
                if token not in chunk.term_freq:
                    continue
                idf = math.log((1 + doc_count) / (1 + self.doc_freq.get(token, 0))) + 1
                score += chunk.term_freq[token] * idf
            if score <= 0:
                continue
            score /= math.sqrt(sum(chunk.term_freq.values()))
            results.append(
                RetrievalHit(
                    source=chunk.source,
                    title=chunk.title,
                    content=chunk.content,
                    score=score,
                    start_line=chunk.start_line,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def search_many(self, queries: Iterable[str], limit: int = 5) -> List[RetrievalHit]:
        merged: Dict[str, RetrievalHit] = {}
        for query in queries:
            for hit in self.search(query, limit=limit):
                key = f"{hit.source}:{hit.start_line}"
                best = merged.get(key)
                if best is None or hit.score > best.score:
                    merged[key] = hit
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return results[:limit]
