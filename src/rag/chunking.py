"""Turns the vLLM markdown docs into retrieval chunks.

Two-stage, header-aware splitting:
  1. split each page at its markdown headings, so a chunk never mixes two sections;
  2. split any section that is still too long on paragraph -> line -> sentence boundaries,
     with overlap so a fact sitting on a boundary appears whole in at least one chunk.
Every chunk records its page title and section heading ("breadcrumb"); the breadcrumb is
prepended before embedding and is what citations point to.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CHARS_PER_TOKEN = 4
HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str          # relative to the corpus root, e.g. "serving/parallelism_scaling.md"
    title: str             # page title (first H1)
    heading: str           # nearest section heading, "" for the page intro
    text: str

    @property
    def breadcrumb(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title

    @property
    def embedding_text(self) -> str:
        return f"{self.breadcrumb}\n\n{self.text}"


def clean_markdown(raw: str) -> str:
    """Removes MkDocs-only syntax that carries no meaning for retrieval."""
    text = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.DOTALL)        # YAML front matter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)               # HTML comments
    lines = []
    for line in text.splitlines():
        if "--8<--" in line:                                              # snippet includes
            continue
        line = re.sub(r"^(\s*)(!!!|\?\?\?\+?)\s*(\w+)\s*\"?(.*?)\"?\s*$",  # admonitions
                      lambda m: f"{m.group(1)}{m.group(3).capitalize()}: {m.group(4)}".rstrip(": "),
                      line)
        line = re.sub(r'^===\s+"(.*)"\s*$', r"\1:", line)                  # content tabs
        line = re.sub(r"</?(details|summary|figure|div|p|br|img)[^>]*>", "", line)
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _split_sections(text: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (page_title, [(heading, body), ...]); ignores '#' lines inside code fences."""
    title, heading, body, sections, in_fence = None, "", [], [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            sections.append((heading, "\n".join(body).strip()))
            level, name = len(match.group(1)), match.group(2).strip()
            if level == 1 and title is None:
                title, heading = name, ""
            else:
                heading = name
            body = []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body).strip()))
    return title or fallback_title, [(h, b) for h, b in sections if b]


def _split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    for separator in ("\n\n", "\n", ". ", " "):
        parts = text.split(separator)
        if len(parts) > 1:
            break
    else:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap_chars)]

    pieces, current = [], ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) > max_chars:
            pieces.extend(_split_long(part, max_chars, overlap_chars))
            current = ""
        else:
            # Carry the tail of the previous piece forward as overlap.
            tail = current[-overlap_chars:].split(" ", 1)[-1] if current else ""
            current = f"{tail}{separator}{part}" if tail else part
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc_path: str, raw: str, chunking: dict) -> list[Chunk]:
    max_chars = chunking["chunk_size_tokens"] * CHARS_PER_TOKEN
    overlap_chars = chunking["chunk_overlap_tokens"] * CHARS_PER_TOKEN
    min_chars = chunking["min_chunk_tokens"] * CHARS_PER_TOKEN

    fallback_title = Path(doc_path).stem.replace("_", " ").title()
    title, sections = _split_sections(clean_markdown(raw), fallback_title)
    chunks = []
    for heading, body in sections:
        for piece in _split_long(body, max_chars, overlap_chars):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            digest = hashlib.sha1(f"{doc_path}|{heading}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, doc_path, title, heading, piece))
    return chunks


def chunk_corpus(corpus_dir: Path, chunking: dict) -> list[Chunk]:
    chunks = []
    for path in sorted(corpus_dir.rglob("*.md")):
        doc_path = path.relative_to(corpus_dir).as_posix()
        chunks.extend(chunk_document(doc_path, path.read_text(encoding="utf-8"), chunking))
    # Identical text can appear on several pages (shared snippets); index it once.
    unique = {chunk.chunk_id: chunk for chunk in chunks}
    return list(unique.values())
