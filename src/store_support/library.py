import hashlib
import math
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from store_support.types import Passage

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "for",
        "is",
        "are",
        "was",
        "be",
        "on",
        "in",
        "it",
        "this",
        "that",
        "with",
        "or",
        "not",
        "from",
        "by",
        "as",
        "at",
        "if",
        "can",
        "you",
        "your",
        "we",
        "our",
        "do",
        "does",
        "what",
        "how",
        "my",
        "me",
        "i",
    }
)


class HashEmbedder:
    """Local retrieval so a run without an OpenAI key still searches the Support library."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            vector[index] += 1.0
        return vector


class OpenAIEmbedder:
    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        embed_documents = getattr(self._embeddings, "embed_documents")
        vectors = embed_documents(texts)
        return [list(vector) for vector in vectors]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Passage, list[float]]] = []

    def add(self, passages: list[Passage], vectors: list[list[float]]) -> None:
        self._items.extend(zip(passages, vectors, strict=True))

    def search(self, vector: list[float], k: int) -> list[tuple[Passage, float]]:
        scored = [(passage, _cosine(vector, stored)) for passage, stored in self._items]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class SupportLibrary:
    def __init__(self, embedder: HashEmbedder | OpenAIEmbedder, min_score: float = 0.12) -> None:
        self._embedder = embedder
        self._store = InMemoryVectorStore()
        self._passages: list[Passage] = []
        self._documents: list[dict[str, str]] = []
        self._min_score = min_score
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)

    def add_document(self, name: str, text: str) -> int:
        chunks = [chunk.strip() for chunk in self._splitter.split_text(text) if chunk.strip()]
        if not chunks:
            return 0
        passages = [Passage(document=name, text=chunk) for chunk in chunks]
        self._documents.append({"name": name, "title": _title(name, text), "text": text.strip()})
        self._passages.extend(passages)
        self._store.add(passages, self._embedder.embed(chunks))
        return len(passages)

    def documents(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(document) for document in self._documents)

    def search(self, query: str, k: int = 3) -> tuple[Passage, ...]:
        if isinstance(self._embedder, HashEmbedder):
            return self._search_lexical(query, k)
        vector = self._embedder.embed([query])[0]
        scored = self._store.search(vector, k=8)
        if not scored or scored[0][1] < self._min_score:
            return ()
        best = scored[0][1]
        kept = [passage for passage, score in scored if score >= best * 0.72]
        return tuple(kept[:k])

    def _search_lexical(self, query: str, k: int) -> tuple[Passage, ...]:
        query_tokens = set(_tokens(query))
        if not query_tokens or not self._passages:
            return ()
        document_frequency: Counter[str] = Counter()
        tokenized = [set(_tokens(passage.text)) for passage in self._passages]
        for tokens in tokenized:
            document_frequency.update(tokens)
        total = len(self._passages)
        scored: list[tuple[Passage, float]] = []
        for passage, tokens in zip(self._passages, tokenized, strict=True):
            weight = 0.0
            for token in query_tokens:
                frequency = document_frequency[token]
                if frequency == 0 or token not in tokens:
                    continue
                if frequency == total and total > 1:
                    continue
                weight += 1.0 + math.log(total / frequency)
            scored.append((passage, weight))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored or scored[0][1] <= 0:
            return ()
        best = scored[0][1]
        kept = [passage for passage, score in scored if score >= best * 0.72 and score > 0]
        return tuple(kept[:k])


def _title(name: str, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return name.rsplit(".", 1)[0].replace("-", " ").capitalize()


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOP or len(raw) < 3:
            continue
        tokens.append(raw)
        if raw.endswith("s") and len(raw) > 4:
            tokens.append(raw[:-1])
    return tokens


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
