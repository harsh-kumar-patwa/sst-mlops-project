"""Prints every doc page with its section headings, as a reference for writing golden questions.

Usage: make docmap   (writes eval/doc_map.md)
The source_docs column of golden.csv must use the paths printed here.
"""

from collections import defaultdict

from rag.chunking import chunk_corpus
from rag.config import REPO_ROOT, load_config


def main() -> None:
    config = load_config()
    sections = defaultdict(list)
    titles = {}
    for chunk in chunk_corpus(REPO_ROOT / config["corpus"]["path"], config["chunking"]):
        titles[chunk.doc_path] = chunk.title
        if chunk.heading and chunk.heading not in sections[chunk.doc_path]:
            sections[chunk.doc_path].append(chunk.heading)

    print(f"# vLLM {config['corpus']['version']} docs: pages and sections\n")
    print("Use the `path` exactly as shown in the `source_docs` column of eval/golden.csv.\n")
    for path in sorted(titles):
        print(f"- `{path}`: **{titles[path]}**")
        for heading in sections[path]:
            print(f"  - {heading}")


if __name__ == "__main__":
    main()
