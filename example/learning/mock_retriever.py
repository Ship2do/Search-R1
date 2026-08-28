"""A tiny retriever implementing the same HTTP contract as Search-R1.

Run from the repository root:

    uvicorn example.learning.mock_retriever:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "example" / "corpus.jsonl"


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


def load_corpus(path: Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as corpus_file:
        return [json.loads(line) for line in corpus_file if line.strip()]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_corpus(
    query: str,
    corpus: list[dict[str, Any]],
    topk: int,
) -> list[tuple[dict[str, Any], float]]:
    """Rank documents by deterministic token overlap for protocol testing."""
    query_terms = _terms(query)
    ranked = []
    for document in corpus:
        overlap = len(query_terms & _terms(document["contents"]))
        score = overlap / max(len(query_terms), 1)
        ranked.append((document, score))
    ranked.sort(key=lambda item: (-item[1], str(item[0].get("id", ""))))
    return ranked[:topk]


CORPUS = load_corpus()
app = FastAPI(title="Search-R1 learning mock retriever")


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> dict[str, list[list[dict[str, Any]]]]:
    topk = request.topk or 3
    results = []
    for query in request.queries:
        ranked = search_corpus(query, CORPUS, topk)
        if request.return_scores:
            results.append([
                {"document": document, "score": score}
                for document, score in ranked
            ])
        else:
            results.append([document for document, _ in ranked])
    return {"result": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
