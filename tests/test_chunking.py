from rag.chunking import chunk_document, clean_markdown

SETTINGS = {"chunk_size_tokens": 50, "chunk_overlap_tokens": 10, "min_chunk_tokens": 2}


def test_clean_markdown_strips_mkdocs_syntax():
    raw = '---\ntitle: x\n---\n# Page\n<!-- hidden -->\n--8<-- "snippet.md"\n!!! note "Careful"\n    body\n=== "Tab A"\n'
    cleaned = clean_markdown(raw)
    assert "title: x" not in cleaned
    assert "hidden" not in cleaned
    assert "--8<--" not in cleaned
    assert "Note: Careful" in cleaned
    assert "Tab A:" in cleaned


def test_headings_inside_code_fences_are_not_sections():
    raw = "# Install\n\nRun this:\n\n```bash\n# not a heading\npip install vllm\n```\n\n## Next\n\nMore text here."
    chunks = chunk_document("x.md", raw, SETTINGS)
    assert [c.heading for c in chunks] == ["", "Next"]
    assert "# not a heading" in chunks[0].text
    assert chunks[0].title == "Install"


def test_long_sections_split_with_overlap_and_keep_breadcrumb():
    sentences = " ".join(f"Sentence number {i} talks about tensor parallelism." for i in range(40))
    chunks = chunk_document("serving/x.md", f"# Scaling\n\n## TP\n\n{sentences}", SETTINGS)
    assert len(chunks) > 1
    assert all(len(c.text) <= SETTINGS["chunk_size_tokens"] * 4 + 50 for c in chunks)
    assert all(c.breadcrumb == "Scaling > TP" for c in chunks)
    assert chunks[1].embedding_text.startswith("Scaling > TP\n\n")
